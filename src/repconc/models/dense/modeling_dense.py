from ast import Pass
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from typing import Union
from transformers import AutoConfig, BertConfig, BertModel, RobertaConfig, RobertaModel, AutoModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel, DistilBertConfig


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BertDense(BertPreTrainedModel):
    def __init__(self, config: BertConfig):
        BertPreTrainedModel.__init__(self, config)
        self.bert = BertModel(config, add_pooling_layer=False)
    
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        if hasattr(self.config, "pooling"):
            if self.config.pooling == "cls":
                text_embeds = outputs.last_hidden_state[:, 0]
            elif self.config.pooling == "mean":
                text_embeds = mean_pooling(outputs, attention_mask)
            else:
                raise NotImplementedError()
        else:
            text_embeds = outputs.last_hidden_state[:, 0]
        if hasattr(self.config, "similarity_metric"):
            if self.config.similarity_metric == "METRIC_IP":
                pass
            elif self.config.similarity_metric == "METRIC_COS":
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            else:
                pass
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds

    @property
    def language_model(self):
        return self.bert


class RobertaDense(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
    
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.roberta(input_ids, attention_mask, return_dict=True)
        text_embeds = outputs.last_hidden_state[:, 0]
        if hasattr(self.config, "pooling"):
            if self.config.pooling == "cls":
                text_embeds = outputs.last_hidden_state[:, 0]
            elif self.config.pooling == "mean":
                text_embeds = mean_pooling(outputs, attention_mask)
            else:
                raise NotImplementedError()
        else: # default: use cls token embedding
            text_embeds = outputs.last_hidden_state[:, 0]
        if hasattr(self.config, "similarity_metric"):
            if self.config.similarity_metric == "METRIC_IP":
                pass
            elif self.config.similarity_metric == "METRIC_COS":
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            else: # default: use the original embedding
                pass
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds

    @property
    def language_model(self):
        return self.roberta


class DistilBertDense(DistilBertPreTrainedModel):
    def __init__(self, config: DistilBertConfig):
        DistilBertPreTrainedModel.__init__(self, config)
        self.distilbert = DistilBertModel(config)
    
    def forward(self, input_ids, attention_mask, return_dict=False):
        outputs = self.distilbert(input_ids, attention_mask, return_dict=True)
        if hasattr(self.config, "pooling"):
            if self.config.pooling == "cls":
                text_embeds = outputs.last_hidden_state[:, 0]
            elif self.config.pooling == "mean":
                text_embeds = mean_pooling(outputs, attention_mask)
            else:
                raise NotImplementedError()
        else:
            text_embeds = outputs.last_hidden_state[:, 0]
        if hasattr(self.config, "similarity_metric"):
            if self.config.similarity_metric == "METRIC_IP":
                pass
            elif self.config.similarity_metric == "METRIC_COS":
                text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            else:
                pass
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds

    @property
    def language_model(self):
        return self.distilbert


class AutoDense:
    @staticmethod
    def from_pretrained(model_name_or_path: str, config = None) -> Union[BertDense, RobertaDense, DistilBertDense]:
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path)
        if config.model_type == "bert":
            model = BertDense.from_pretrained(model_name_or_path, config=config)
        elif config.model_type == "roberta":
            model = RobertaDense.from_pretrained(model_name_or_path, config=config)
        elif config.model_type == "distilbert":
            model = DistilBertDense.from_pretrained(model_name_or_path, config=config)
        else:
            raise NotImplementedError()
        return model