import os
import json
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import negative, nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Union, List, Dict, Any
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, TrainingArguments
from transformers import TrainingArguments, Trainer
from torch.cuda.amp import autocast
from transformers.integrations import TrainerCallback
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.modeling_utils import unwrap_model
if is_sagemaker_mp_enabled():
    raise NotImplementedError()
    import smdistributed.modelparallel.torch as smp
from grad_cache import GradCache

from .modeling_repconc import RepCONC, QuantizeOutput
logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    qrel_path: str = field()
    query_path: str = field()
    corpus_path: str = field()  
    valid_qrel_path: str = field()
    valid_query_path: str = field()
    valid_corpus_path: str = field()  
    max_query_len: int = field()
    max_doc_len: int = field()  


@dataclass
class RepCONCFinetuneArguments(TrainingArguments):
    negative_per_query: int = field(default=1)
    dynamic_topk_hard_negative: int = field(default=None)
    centroid_learning_rate: float = field(default=1e-3)
    temperature: float = field(default=1.0)
    mse_loss_weight: float = field(default=0)
    not_use_constraint: bool = field(default=False)
    negative: str = field(default="random", metadata={"help": "inbatch or random or path to a json file."})
    cache_chunk_size: int = field(default=-1)
    seed: int = field(default=2022)

    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

    
@dataclass
class FinetuneCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_query_len: int, max_doc_len: int, ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # tokenizing batch of text is much faster
        query_input = self.tokenizer(
            [x['query'] for x in features],
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_query_len
        )
        pos_doc_input = self.tokenizer(
            [x['pos_doc'] for x in features],
            padding=True,
            return_tensors='pt',
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_doc_len
        )
        # we have to prevent inbatch false negatives when gathering tensors in the trainer
        # because each distributed process has its own collators
        qids = torch.tensor([x['qid'] for x in features], dtype=torch.long)
        pos_docids = torch.tensor([x['pos_docid'] for x in features], dtype=torch.long)

        batch_data = {
                "query_input": query_input,
                "pos_doc_input": pos_doc_input,
                "qids": qids,
                "pos_docids": pos_docids,
            }

        if "neg_docs" in features[0]:
            neg_doc_input = self.tokenizer(
                sum([x['neg_docs'] for x in features], []),
                padding=True,
                return_tensors='pt',
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=self.max_doc_len
            )     
            neg_docids = torch.tensor(sum([x['neg_docids'] for x in features], []), dtype=torch.long)  
            batch_data.update({
                "neg_doc_input": neg_doc_input,
                "neg_docids": neg_docids,
            }) 
        return batch_data


class QDRelDataset(Dataset):
    def __init__(self, 
            tokenizer: PreTrainedTokenizer, 
            qrel_path: str, 
            query_path: str, 
            corpus_path: str, 
            max_query_len: int, 
            max_doc_len: int, 
            negative: str, 
            negative_per_query: int,
            rel_threshold=1, 
            verbose=True):
        '''
        negative: choices from `inbatch', `random', or a path to a json file that contains \
            the qid:neg_pid_lst  
        '''
        super().__init__()
        self.tokenizer = tokenizer
        self.queries, qid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(query_path), disable=not verbose, mininterval=10)):
            qid, query = line.split("\t")
            qid2offset[qid] = idx
            self.queries.append(query.strip())

        self.corpus, docid2offset = [], dict()
        for idx, line in enumerate(tqdm(open(corpus_path), disable=not verbose, mininterval=10)):
            splits = line.strip().split("\t")
            docid, text_fields = splits[0], splits[1:]
            text = f'{tokenizer.sep_token}'.join((t.strip() for t in text_fields))
            docid2offset[docid] = idx
            self.corpus.append(text.strip()[:10000])

        self.qrels = defaultdict(list)
        for line in tqdm(open(qrel_path), disable=not verbose, mininterval=10):
            qid, _, docid, rel = line.split()
            if int(rel) >= rel_threshold:
                qoffset = qid2offset[qid]
                docoffset = docid2offset[docid]
                self.qrels[qoffset].append(docoffset)

        self.negative_per_query = negative_per_query
        if negative not in ["inbatch", "random"]:
            self.negative = {}
            for qid, docid_lst in tqdm(json.load(open(negative)).items(), disable=not verbose, mininterval=10):
                self.negative[qid2offset[qid]] = [docid2offset[docid] for docid in docid_lst]
        else:
            self.negative = negative

        self.qids = sorted(self.qrels.keys())
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.qrels = dict(self.qrels)

    def get_qrels(self):
        return self.qrels

    def __len__(self):
        return len(self.qids)
    
    def __getitem__(self, index):
        '''
        We do not tokenize text here and instead tokenize batch of text in the collator because
            a. Tokenizing batch of text is much faster then tokenizing one by one
            b. Usually, the corpus is too large and we cannot afford to use multiple num workers
        '''
        qid = self.qids[index]
        query = self.queries[qid]
        pos_docids = self.qrels[qid]
        pos_docid = random.choice(pos_docids)
        pos_doc = self.corpus[pos_docid]
        data = {
            "query": query,
            "pos_doc": pos_doc,
            "pos_docid": pos_docid,
            "qid": qid
        }
        if self.negative == "inbatch":
            assert self.negative_per_query == 0
        else:
            if self.negative == "random":
                neg_docids = random.sample(range(len(self.corpus)), self.negative_per_query)
            else:
                neg_docids = random.sample(self.negative[qid], self.negative_per_query)
            neg_docs = [self.corpus[neg_docid] for neg_docid in neg_docids]
            data.update({"neg_docids": neg_docids, "neg_docs": neg_docs})
        return data


class RepCONC_Norm_Centroid_Callback(TrainerCallback):
    def on_step_end(self, *args, **kwargs):
        """
        Event called at the end of a training step. Normalize centroids if necessary.
        """
        model: RepCONC = kwargs.pop('model')
        unwrap_model(model).normalize_centrodis()


class RepCONCFinetuner(Trainer):
    def __init__(self, qrels, *args, **kwargs):
        logger.info('Initializing Gradient Cache Trainer')
        super(RepCONCFinetuner, self).__init__(*args, **kwargs)
        
        self.qrels = qrels # is used to compute negative mask
        self.args: RepCONCFinetuneArguments
        self.model: RepCONC
        if self.args.cache_chunk_size != -1:
            self.gc = GradCache(
                models=[self.model],
                chunk_sizes=self.args.cache_chunk_size,
                loss_fn=self.compute_contrastive_loss,
                get_rep_fn=lambda x: x.continuous_embeds, 
                fp16=self.args.fp16,
                scaler=self.scaler
            )
        if self.model.config.similarity_metric == "METRIC_CENTROID_COS":
            self.add_callback(RepCONC_Norm_Centroid_Callback)
    
    def training_step(self, model: RepCONC, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.cache_chunk_size == -1:
            raise NotImplementedError()
        if self.args.gradient_accumulation_steps > 1:
            raise ValueError
        model.train()
        query_input, pos_doc_input = inputs['query_input'], inputs['pos_doc_input']
        query_input = self._prepare_inputs(query_input)
        pos_doc_input = self._prepare_inputs(pos_doc_input)
        query_chunked_inputs = self.split_tensor_dict(query_input)
        pos_doc_chunked_inputs = self.split_tensor_dict(pos_doc_input)

        if "neg_doc_input" in inputs:
            neg_doc_input = self._prepare_inputs(inputs['neg_doc_input'])
            neg_doc_chunked_inputs = self.split_tensor_dict(neg_doc_input)
            total_contrast_loss, query_grad_cache, query_rnd_states, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc, neg_doc_grad_cache, neg_doc_rnd_states, codes_neg_doc= \
                self._build_grad_cache_for_explicit_neg(model, query_chunked_inputs, pos_doc_chunked_inputs, neg_doc_chunked_inputs, inputs['qids'], inputs['pos_docids'], inputs['neg_docids'])
            self._forward_backward(model,
                [
                    [query_chunked_inputs, query_grad_cache, query_rnd_states, None], 
                    [pos_doc_chunked_inputs, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc],
                    [neg_doc_chunked_inputs, neg_doc_grad_cache, neg_doc_rnd_states, codes_neg_doc],
                ])
        else:
            total_contrast_loss, query_grad_cache, query_rnd_states, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc = \
                self._build_grad_cache_for_inbatch_neg(model, query_chunked_inputs, pos_doc_chunked_inputs, inputs['qids'], inputs['pos_docids'])
            self._forward_backward(model,
                [
                    [query_chunked_inputs, query_grad_cache, query_rnd_states, None], 
                    [pos_doc_chunked_inputs, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc]
                ])
        return torch.tensor(total_contrast_loss)

    def _build_grad_cache_for_inbatch_neg(self, model: RepCONC, 
            query_chunked_inputs,
            pos_doc_chunked_inputs, 
            qids, docids
        ):
        query_embeds, query_rnd_states = self.gc.forward_no_grad(model, query_chunked_inputs)  
        pos_doc_embeds, pos_doc_rnd_states = self.gc.forward_no_grad(model, pos_doc_chunked_inputs)  

        # In distributed training, RepCONC benefits from large batch
        # RepCONC will automatically use distributed training if it is enabled.
        codes_pos_doc = unwrap_model(model).quantize(pos_doc_embeds)
        quantized_pos_doc_embeds = unwrap_model(model).decode(codes_pos_doc)

        if self.state.global_step % self.args.logging_steps == 0:
            quant_states = test_quantize(pos_doc_embeds, self.model, self.args.local_rank, block_id=0)
            self.log(quant_states)

        qids = self._prepare_input(qids).contiguous() # _prepare_input to gpu
        docids = self._prepare_input(docids).contiguous()        
        if self.args.local_rank > -1:
            query_embeds = self.gather_tensors(query_embeds.contiguous())[0]
            quantized_pos_doc_embeds = self.gather_tensors(quantized_pos_doc_embeds.contiguous())[0]   
            qids, docids = self.gather_tensors(qids, docids)      

        (query_grad_cache, pos_doc_grad_cache), total_loss = \
            self.gc.build_cache(query_embeds, quantized_pos_doc_embeds, qids=qids, docids=docids)
        return total_loss.item(), query_grad_cache, query_rnd_states, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc

    def _build_grad_cache_for_explicit_neg(self, model: RepCONC, 
            query_chunked_inputs,
            pos_doc_chunked_inputs, 
            neg_chunked_inputs,
            qids, docids, neg_docids
        ):
        query_embeds, query_rnd_states = self.gc.forward_no_grad(model, query_chunked_inputs)  
        pos_doc_embeds, pos_doc_rnd_states = self.gc.forward_no_grad(model, pos_doc_chunked_inputs)  
        neg_doc_embeds, neg_doc_rnd_states = self.gc.forward_no_grad(model, neg_chunked_inputs)  
        
        # In distributed training, RepCONC benefits from large batch
        cat_pos_neg_doc_embeds = torch.vstack((pos_doc_embeds, neg_doc_embeds))
        cat_codes_doc = unwrap_model(model).quantize(cat_pos_neg_doc_embeds)
        codes_pos_doc = cat_codes_doc[:len(pos_doc_embeds)]
        codes_neg_doc = cat_codes_doc[len(pos_doc_embeds):]
        quantized_pos_doc_embeds = unwrap_model(model).decode(codes_pos_doc)
        quantized_neg_doc_embeds = unwrap_model(model).decode(codes_neg_doc)

        if self.state.global_step % self.args.logging_steps == 0:
            quant_states = test_quantize(cat_pos_neg_doc_embeds, self.model, self.args.local_rank, block_id=0)
            self.log(quant_states)

        qids = self._prepare_input(qids).contiguous() # _prepare_input to gpu
        docids = self._prepare_input(docids).contiguous()   
        neg_docids = self._prepare_input(neg_docids).contiguous()   
        if self.args.local_rank > -1:
            query_embeds = self.gather_tensors(query_embeds.contiguous())[0]
            quantized_pos_doc_embeds = self.gather_tensors(quantized_pos_doc_embeds.contiguous())[0]
            quantized_neg_doc_embeds = self.gather_tensors(quantized_neg_doc_embeds.contiguous())[0]
            qids, docids, neg_docids = self.gather_tensors(qids, docids, neg_docids)  

        all_quantized_doc_embeds = torch.vstack((quantized_pos_doc_embeds, quantized_neg_doc_embeds))
        all_docids = torch.hstack((docids, neg_docids))

        (query_grad_cache, all_doc_grad_cache), total_loss = \
            self.gc.build_cache(query_embeds, all_quantized_doc_embeds, qids=qids, docids=all_docids)
        pos_doc_grad_cache, neg_doc_grad_cache = all_doc_grad_cache[:len(quantized_pos_doc_embeds)], all_doc_grad_cache[len(quantized_pos_doc_embeds):]
        assert len(neg_doc_grad_cache) == len(quantized_neg_doc_embeds), f"{neg_doc_grad_cache.shape}, {quantized_neg_doc_embeds.shape}"
        return total_loss.item(), query_grad_cache, query_rnd_states, pos_doc_grad_cache, pos_doc_rnd_states, codes_pos_doc, neg_doc_grad_cache, neg_doc_rnd_states, codes_neg_doc

    def _forward_backward(self, model: RepCONC, lst_chunked_inputs_cache_rnd):
        '''
        lst_chunked_inputs_cache_rnd: list of [chunked_inputs, grad_cache, rnd_states, discrete_codes (for query it is None)]
        '''        
        for chunk_input_id, (chunked_inputs, grad_cache, rnd_states, discrete_codes) in enumerate(lst_chunked_inputs_cache_rnd):
            for local_chunk_id, chunk in enumerate(chunked_inputs):
                assert self.args.local_rank < 0 or len(chunked_inputs) * self.args.cache_chunk_size * dist.get_world_size() == len(grad_cache), \
                    f"{len(chunked_inputs)} * {self.args.cache_chunk_size} * {dist.get_world_size()} == {len(grad_cache)}"
                device_offset = 0 if self.args.local_rank < 0 else self.args.cache_chunk_size * len(chunked_inputs) * self.args.local_rank
                local_offset = local_chunk_id * self.args.cache_chunk_size
                chunk_offset = device_offset + local_offset
                with rnd_states[local_chunk_id]:
                    with autocast(enabled=self.use_amp):
                        if discrete_codes is not None:
                            quant_outputs: QuantizeOutput = model(
                                discrete_codes=discrete_codes[local_offset: local_offset + self.args.cache_chunk_size], 
                                return_quantized_embedding=True,
                                **chunk)
                        else:
                            quant_outputs: QuantizeOutput = model(**chunk)
                        # grad_cache is comptued across device
                        cached_grads = grad_cache[chunk_offset: chunk_offset + quant_outputs.continuous_embeds.size(0)]
                        surrogate = torch.dot(cached_grads.flatten(), quant_outputs.continuous_embeds.flatten())
                        if discrete_codes is not None: # is doc
                            # discrete_codes are only for the embeds on the current device
                            # no need to add device_offset
                            quant_embeds = quant_outputs.quantized_embeds
                            surrogate = surrogate + torch.dot(cached_grads.flatten(), quant_embeds.flatten())
                            mse_loss = ((quant_embeds - quant_outputs.continuous_embeds)**2).sum(-1).mean() * self.args.mse_loss_weight
                    
                ddp_no_sync = self.args.local_rank > -1 and (
                    (local_chunk_id + 1 < len(chunked_inputs)) or (chunk_input_id + 1 < len(lst_chunked_inputs_cache_rnd)))
                with model.no_sync() if ddp_no_sync else nullcontext():
                    if discrete_codes is None: # is query 
                        if self.use_amp:
                            surrogate.backward()
                        elif self.use_apex:
                            raise ValueError
                        elif self.deepspeed:
                            raise ValueError
                        else:
                            surrogate.backward()
                    else: # is doc
                        if self.use_amp:
                            (self.scaler.scale(mse_loss) + surrogate).backward()
                        elif self.use_apex:
                            raise ValueError
                        elif self.deepspeed:
                            raise ValueError
                        else:
                            (mse_loss + surrogate).backward()

    def compute_contrastive_loss(self, query_embeds, doc_embeds, qids, docids):  
        '''
        doc_embeds can be the concatenation of relevant document embeddings and negative document embeddings
        labels are set along the diagnal
        '''
        effective_bsz = self.args.per_device_train_batch_size
        if self.args.local_rank > -1:
            effective_bsz *= dist.get_world_size()
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        negative_mask = torch.logical_or(
            self._compute_mask_for_false_negative(qids, docids), 
            self._compute_mask_for_duplicate_negative(qids, docids)
        )
        negative_mask = negative_mask.type(torch.float32)
        similarities = torch.matmul(query_embeds, doc_embeds.transpose(0, 1))

        if self.model.config.similarity_metric == "METRIC_CENTROID_COS":
            # so that the similarity scores are normalized to -1 -- 1
            # therefore, the hyper-parameters may be consistent among different M values.
            similarities = similarities / self.model.config.MCQ_M
        if self.args.temperature != 1:
            similarities = similarities / self.args.temperature
        similarities = similarities - 10000.0 * negative_mask

        if self.args.dynamic_topk_hard_negative is not None and self.args.dynamic_topk_hard_negative > 0:
            hardneg_mask = torch.ones_like(similarities)
            neg_similarities = similarities.detach().clone()
            # Otherwise, It is possible that the positives are regarded as the dynamic_topk_hard_negative
            neg_similarities.scatter_(1, labels[:, None], -10000.0)            
            hardneg_mask.scatter_(1, torch.topk(neg_similarities, self.args.dynamic_topk_hard_negative).indices, 0)
            hardneg_mask.scatter_(1, torch.arange(similarities.size(0), device=similarities.device)[:, None], 0)
            similarities = similarities - 10000.0 * hardneg_mask
        co_loss = F.cross_entropy(similarities, labels) 
        return co_loss

    @torch.no_grad()
    def _compute_mask_for_false_negative(self, qids, docids):
        negative_mask = torch.zeros((len(qids), len(docids)), dtype=torch.bool, device=qids.device)
        for i, qid in enumerate(qids):
            for d in self.qrels[qid.item()]:
                negative_mask[i] = torch.logical_or(negative_mask[i], docids==d)
        negative_mask.fill_diagonal_(False)
        return negative_mask

    @torch.no_grad()
    def _compute_mask_for_duplicate_negative(self, qids, docids):
        '''
        Remove duplicates in negatives
        '''
        relation_mat = docids[:, None] == docids[None, :]
        negative_mask = torch.triu(relation_mat, diagonal=1).any(dim=0, keepdim=True)
        negative_mask = negative_mask.repeat(len(qids), 1)
        negative_mask.fill_diagonal_(False)
        return negative_mask

    def _gather_tensor(self, t: Tensor):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[self.args.local_rank] = t
        return all_tensors

    def gather_tensors(self, *tt: Tensor):
        tt = [torch.cat(self._gather_tensor(t)) for t in tt]
        return tt

    def split_tensor_dict(self, td: Dict[str, Tensor]):
        keys = list(td.keys())
        chunked_tensors = [td[k].split(self.args.cache_chunk_size) for k in keys]
        return [dict(zip(keys, tt)) for tt in zip(*chunked_tensors)]

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        return 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.config.save_pretrained(output_dir)
        self.model.dense_encoder.save_pretrained(os.path.join(output_dir, 'dense_encoder'))

    def create_optimizer(self):
        from transformers.trainer_pt_utils import (get_parameter_names,)
        from transformers.trainer_utils import ShardedDDPOption
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and "centroids" not in n],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and "centroids" not in n],
                    "weight_decay": 0.0,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if "centroids" in n],
                    "weight_decay": 0.0,
                    'lr': self.args.centroid_learning_rate
                },
            ]
            logger.info(f"optimizer_grouped_parameters: {[len(x['params']) for x in optimizer_grouped_parameters]}")
            logger.info(f"named_parameters: {len(list(self.model.named_parameters()))}")
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                from fairscale.optim import OSS
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes
                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def evaluate(self, 
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",) -> Dict[str, float]:
        torch.cuda.empty_cache()
        from .evaluate_repconc import encode_corpus, encode_query, batch_search, from_pq_to_ivfpq, load_index_to_gpu
        from repconc.utils.eval_utils import pytrec_evaluate

        corpus, queries, qrels = self.eval_dataset
        original_quantize_parameter = self.model.use_constraint
        self.model.use_constraint = False # use nearest neighbor to encode
        fp16, bf16 = self.args.fp16, self.args.bf16
        self.args.fp16, self.args.bf16 = False, False
        dataloader_drop_last = self.args.dataloader_drop_last
        self.args.dataloader_drop_last = False
        index, corpus_ids = encode_corpus(corpus, self.model, self.tokenizer, 512, self.args)
        query_embeds, query_ids = encode_query(queries, self.model, self.tokenizer, 512, self.args)
        self.model.use_constraint = original_quantize_parameter # set back training parameter
        self.args.fp16, self.args.bf16 = fp16, bf16
        self.args.dataloader_drop_last = dataloader_drop_last

        torch.cuda.empty_cache()
        index = from_pq_to_ivfpq(index) # only gpuivfpq is supported
        index = load_index_to_gpu(index, 0 if self.args.local_rank < 0 else self.args.local_rank)
        all_topk_scores, all_topk_ids = batch_search(
            query_ids, query_embeds.astype(np.float32), # fp16
            corpus_ids, index, 
            topk=10, 
            batch_size=512)

        run_results = defaultdict(dict)
        for qid, topk_scores, topk_ids in zip(query_ids, all_topk_scores, all_topk_ids):
            for i, (score, docid) in enumerate(zip(topk_scores, topk_ids)):
                run_results[qid.item()][docid.item()] = score.item()
        metrics = {}
        for category, cat_metrics in pytrec_evaluate(
                qrels, 
                dict(run_results), 
                k_values =(10, ),
                mrr_k_values = (10, ),).items():
            if category == "perquery":
                continue
            for metric, score in cat_metrics.items():
                metrics[f"{metric_key_prefix}_{metric}"] = score
        torch.cuda.empty_cache()
        self.log(metrics)
       
        return metrics


def eval_balance(codes, distributed, block_id: int):
    # we can check the distribution of codes
    if distributed:
        all_codes = [torch.empty_like(codes) for _ in range(dist.get_world_size())]
        dist.all_gather(all_codes, codes)
        all_codes = torch.cat(all_codes, dim=0)
    else:
        all_codes = codes
    invest_codes = all_codes[:, block_id]
    balance_lst = []
    for i in range(256):
        num = (invest_codes == i).sum().item()
        balance_lst.append(abs(1 - num / (len(invest_codes) / 256)))
    average_balance = np.mean(balance_lst).item()
    return {
        "avg_imbalance": round(average_balance, 3),
        "max_imbalance": round(np.max(balance_lst).item(), 3),
    }


@torch.no_grad()
def test_quantize(continuous_embeddings, lm: RepCONC, local_rank, block_id=0):
    states = {}
    original_quantize_parameter = lm.use_constraint
    for prefix, use_constraint in zip(["w/o_conc", "w/_conc"], [False, True]):
        lm.use_constraint = use_constraint
        codes = lm.quantize(continuous_embeddings)
        quantized_cls = lm.decode(codes)
        mse = ((quantized_cls - continuous_embeddings)**2).sum(-1).sqrt().mean()
        states[f"{prefix}_mse"] = round(mse.item(), 3)
        balance_states = eval_balance(codes, local_rank > -1, block_id=block_id)
        states.update({f"{prefix}_{k}": v for k, v in balance_states.items()})
    lm.use_constraint = original_quantize_parameter
    return states