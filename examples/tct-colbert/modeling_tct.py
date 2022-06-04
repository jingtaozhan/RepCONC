import os
import torch
from typing import List, Tuple

from torch import nn
from transformers import BertConfig, BertModel, BertPreTrainedModel, BertTokenizerFast, AutoTokenizer

from repconc.models.repconc import RepCONC

class TCTEncoder(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        BertPreTrainedModel.__init__(self, config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.config.pooling = "mean"
        self.config.similarity_metric = "METRIC_IP"
    
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        token_embeds = outputs.last_hidden_state[:, 4:, :]
        input_mask_expanded = attention_mask[:, 4:].unsqueeze(-1).expand(token_embeds.size()).float()
        text_embeds = torch.sum(token_embeds * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds

    @property
    def language_model(self):
        return self.bert


def tct_repconc_from_pretrained(load_dir, use_constraint, sk_epsilon, sk_iters):
    dense_encoder = TCTEncoder.from_pretrained(os.path.join(load_dir, 'dense_encoder'))
    repconc = RepCONC(
        dense_encoder.config, 
        dense_encoder, 
        use_constraint=use_constraint, 
        sk_epsilon=sk_epsilon, 
        sk_iters=sk_iters)
    repconc.load_state_dict(torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location="cpu"))
    return repconc


class TCTTokenizerFast(BertTokenizerFast):
    '''
    ANCE lowers text before tokenization
    '''
    def __call__(self, text, input_text_type, max_length=None, add_special_tokens=False, **kwargs):
        # TCT does not add special tokens and expands queries to a fixed length
        if input_text_type == "query":
            max_length = 36
            text = ['[CLS] [Q] ' + query + '[MASK]' * 36 for query in text ]
        elif input_text_type == "doc":
            text = ['[CLS] [D] ' + doc for doc in text ]
        else:
            raise NotImplementedError()
        return super().__call__(text, max_length=max_length, add_special_tokens=False, **kwargs)


if __name__ == "__main__":
    print(AutoTokenizer.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco"))
    print("Test tokenizer")
    import inspect

    for tokenizer_class in [AutoTokenizer, TCTTokenizerFast]:
        tokenizer = tokenizer_class.from_pretrained("castorini/tct_colbert-v2-hnp-msmarco")
        print(tokenizer.__class__, tokenizer)
        text_lst = ["I am TCT tokenizer"]

        input_text_type = {"input_text_type": "doc"} if "input_text_type" in inspect.getfullargspec(tokenizer.__call__)[0] else {}
        print(tokenizer.convert_ids_to_tokens(tokenizer(
            text_lst, 
            add_special_tokens=True, 
            max_length=36, 
            truncation=True, **input_text_type)['input_ids'][0]))
        input_text_type = {"input_text_type": "query"} if "input_text_type" in inspect.getfullargspec(tokenizer.__call__)[0] else {}
        print(tokenizer.convert_ids_to_tokens(tokenizer(text_lst, add_special_tokens=True, max_length=36, truncation=True, **input_text_type)['input_ids'][0]))
