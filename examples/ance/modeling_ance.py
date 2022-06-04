import os
import torch
from typing import List, Tuple

from torch import nn
from transformers import AutoTokenizer, RobertaModel, RobertaConfig, RobertaTokenizerFast
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from repconc.models.repconc import RepCONC

class ANCEEncoder(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.config.pooling = "cls"
        self.config.similarity_metric = "METRIC_IP"
    
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.roberta(input_ids, attention_mask, return_dict=True)
        text_embeds = self.norm(self.embeddingHead(outputs.last_hidden_state[:, 0]))
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds

    @property
    def language_model(self):
        return self.roberta


def ance_repconc_from_pretrained(load_dir, use_constraint, sk_epsilon, sk_iters):
    dense_encoder = ANCEEncoder.from_pretrained(os.path.join(load_dir, 'dense_encoder'))
    repconc = RepCONC(
        dense_encoder.config, 
        dense_encoder, 
        use_constraint=use_constraint, 
        sk_epsilon=sk_epsilon, 
        sk_iters=sk_iters)
    repconc.load_state_dict(torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location="cpu"))
    return repconc


class ANCETokenizerFast(RobertaTokenizerFast):
    '''
    ANCE lowers text before tokenization
    '''
    def __call__(self, text, *args, **kwargs):
        assert isinstance(text, List) or isinstance(text, Tuple), \
            f"ANCETokenizer only supports List[str] inputs. Current Input is {text}"
        text = [t.lower() for t in text]
        return super().__call__(text, *args, **kwargs)


if __name__ == "__main__":
    print(AutoTokenizer.from_pretrained("castorini/ance-msmarco-passage"))
    print("Test tokenizer")
    tokenizer = ANCETokenizerFast.from_pretrained("castorini/ance-msmarco-passage")
    print(tokenizer)
    text_lst = ["I am ANCE tokenizer"]
    print(tokenizer(text_lst))
    print(tokenizer([t.lower() for t in text_lst]))
