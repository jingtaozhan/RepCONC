import sys
import os
import torch
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import (
    RobertaModel, RobertaConfig
)
import torch.distributed as dist
from torch import Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast        


class RobertaDot(RobertaPreTrainedModel):
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(self._init_weights)        
    
    def encode(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0]
        embeds = self.norm(self.embeddingHead(full_emb))
        return embeds
    
    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)


class QuantDot(RobertaDot):
    def __init__(self, config:RobertaConfig):
        RobertaDot.__init__(self, config)
        self.centroids = nn.Parameter(torch.zeros((config.MCQ_M, config.MCQ_K, config.hidden_size // config.MCQ_M)))
        self.centroids.data.normal_(mean=0.0, std=config.initializer_range)
        self.centroids.requires_grad = True
        self.apply(self._init_weights)
        # Init rotation with OPQ in the training script
        self.rotation = nn.Parameter(torch.eye(config.hidden_size))
        self.rotation.requires_grad = False
    
    def rotate_encode(self, input_ids, attention_mask):
        # OPQ rotates the output embeddings
        unrotate_embeds = self.encode(input_ids, attention_mask)
        self.rotation = self.rotation.to(unrotate_embeds.device)
        embeds = unrotate_embeds @ self.rotation.T
        return embeds

    def forward(self, input_ids, attention_mask):
        return self.rotate_encode(input_ids, attention_mask)

    def batch_rotate_encode(self, input_ids, attention_mask, batch_num):
        # batch encoding can reduce the peak CUDA memory when training with grad checkpointing
        outputs = []
        end = 0
        for i in range(1, batch_num+1):
            start, end = end, int(i/batch_num*len(input_ids))
            out = self.rotate_encode(
                input_ids[start:end], attention_mask[start:end]
            )
            outputs.append(out)
        cat_outputs = torch.vstack(outputs)
        assert len(cat_outputs) == len(input_ids)
        return cat_outputs


class QuantDot_STAR(QuantDot):
    def __init__(self, config:RobertaConfig):
        QuantDot.__init__(self, config)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.apply(self._init_weights)
        self.cluster_distrib = torch.empty((config.MCQ_M, config.MCQ_K)).fill_(1/config.MCQ_K)

    def forward(self, input_query_ids, query_attention_mask,
            input_doc_ids, doc_attention_mask, 
            scores_mask, labels):

        query_embeds = self.rotate_encode(input_query_ids, query_attention_mask)
        doc_embeds = self.batch_rotate_encode(input_doc_ids, doc_attention_mask, 
            self.config.small_batch_num)
        doc_bs = len(doc_embeds)        
        
        #-----------comupte distances with centroids----
        with torch.no_grad():
            doc_embeds = doc_embeds.reshape(
                    len(doc_embeds), self.config.MCQ_M, 1, -1)
            doc_embeds = doc_embeds.transpose(0,1) # M, bs', 1, subdim
            
            expanded_distances = ((doc_embeds - self.centroids.unsqueeze(1))**2).sum(-1) # M, bs', K
            # Maybe center the distance values can help alleviate the nan/inf problem 
            # when using sinkhorn algorithm
            mean_expanded_distances = expanded_distances.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            centered_expanded_distances = expanded_distances - mean_expanded_distances

        #-----------computing codes with sinkhorn algorithm
        self.cluster_distrib = self.cluster_distrib.to(centered_expanded_distances.device)
        expanded_Q = sinkhorn_algorithm(
            -centered_expanded_distances.transpose(1, 2), self.config.sk_epsilon, 
            self.config.sk_iters, self.cluster_distrib, 
            use_distrib_train=False
        ).transpose(1,2) # M-B-K
        Q = expanded_Q[:, :doc_bs, :]
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            codes = torch.argmin(centered_expanded_distances[:, :doc_bs, :], dim=-1) # M, bs
        else:
            codes = torch.argmax(Q, dim=-1) # M, bs
        codes = codes.t()

        #-----------compute quantized embeddings--------
        M = self.centroids.shape[0]
        first_indices = torch.arange(M).to(codes.device)
        first_indices = first_indices.expand(doc_bs, M).reshape(-1)
        quant_doc_embeds = self.centroids[first_indices, codes.reshape(-1)].reshape(doc_bs, -1)

        #-----------mix embeds using detach trick-------
        mixed_doc_embeds = quant_doc_embeds + doc_embeds - doc_embeds.detach()
        
        #-----------computing loss---------------------
        all_doc_embeds = mixed_doc_embeds
        scores = torch.matmul(query_embeds, all_doc_embeds.T) / self.config.loss_temperature
        scores -= scores_mask * 1e4
        rank_loss = self.cross_entropy(scores, labels)

        #----------aggregate rank/mse loss--------------
    
        mse_loss = ((quant_doc_embeds - doc_embeds)**2).sum(-1).mean()
        loss = rank_loss + mse_loss * self.config.mse_weight
    
        return (loss, rank_loss.detach(), mse_loss.detach())


@torch.no_grad()
def sinkhorn_algorithm(out:Tensor, epsilon:float, 
        sinkhorn_iterations:int, cluster_distrib:Tensor, 
        use_distrib_train:bool):
    Q = torch.exp(out / epsilon) # Q is M-K-by-B

    M = Q.shape[0]
    B = Q.shape[2] # number of samples to assign
    K = Q.shape[1] # how many prototypes

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    if use_distrib_train:
        B *= dist.get_world_size()
        dist.all_reduce(sum_Q)
    Q /= sum_Q
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=2, keepdim=True)
        if use_distrib_train:
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        # Q /= K
        Q *= cluster_distrib[:, :, None]

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q