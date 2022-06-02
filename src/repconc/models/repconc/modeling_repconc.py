import os
import logging
import torch
import numpy as np
from torch import nn, Tensor
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput


from repconc.models.dense import BertDense, RobertaDense
from repconc.models.dense.modeling_dense import AutoDense
logger = logging.getLogger(__name__)


@dataclass
class QuantizeOutput(ModelOutput):
    continuous_embeds: Optional[torch.FloatTensor] = None
    quantized_embeds: Optional[torch.FloatTensor] = None
    discrete_codes: Optional[torch.LongTensor] = None


class RepCONC(nn.Module):

    def __init__(self, 
            config: PretrainedConfig, 
            dense_encoder: Union[BertDense, RobertaDense], 
            use_constraint: bool,
            sk_epsilon: float, 
            sk_iters: int):
        super().__init__()
        self.config = config
        self.dense_encoder = dense_encoder
        # so we can use the rotation matrix of OPQ
        self.register_buffer('rotation', torch.eye(dense_encoder.config.hidden_size))
        self.centroids = nn.Parameter(torch.randn((config.MCQ_M, config.MCQ_K, config.hidden_size // config.MCQ_M)))
        if self.config.similarity_metric == "METRIC_CENTROID_COS":
            self.normalize_centrodis()
        self.centroids.requires_grad = True
        self.use_constraint, self.sk_epsilon, self.sk_iters = use_constraint, sk_epsilon, sk_iters

    @torch.no_grad()
    def quantize(self, continuous_embeds):
        batchsize_per_device = len(continuous_embeds)
        distances = ((continuous_embeds.reshape(batchsize_per_device, self.config.MCQ_M, 1, -1).transpose(0,1) - self.centroids.unsqueeze(1))**2).sum(-1) # M, bs', K
        if not self.use_constraint:
            codes = torch.argmin(distances, dim=-1) # M, bs
        else:
            distances = self.center_distance_for_constraint(distances) # to stablize
            # avoid nan
            distances = distances.double()
            Q = sinkhorn_algorithm(
                -distances.transpose(1, 2), 
                self.sk_epsilon, 
                self.sk_iters, 
                use_distrib_train=dist.is_initialized()
            ).transpose(1,2) # M-B-K
            codes = torch.argmax(Q, dim=-1)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                logger.warning(f"Sinkhorn Algorithm returns nan/inf values.")
        codes = codes.t() # bs, M
        return codes
            
    def decode(self, codes):
        # codes: bs, M
        return decode(codes, self.centroids)

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: M, bs, K
        max_distance = distances.max(-1).values.max(-1).values
        min_distance = distances.min(-1).values.min(-1).values
        if dist.is_initialized():
            dist.all_reduce(max_distance, torch.distributed.ReduceOp.MAX)
            dist.all_reduce(min_distance, torch.distributed.ReduceOp.MIN)
        middle = (max_distance + min_distance)/2
        amplitude = max_distance - middle + 1e-5
        assert torch.all(amplitude > 0)
        centered_distances = (distances - middle[:, None, None]) / amplitude[:, None, None]
        return centered_distances
        
    def forward(self, 
            input_ids, 
            attention_mask, 
            discrete_codes = None,
            return_code = False,
            return_quantized_embedding = False
        ):
        dense_embed = self.dense_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        rotated_embed = dense_embed @ self.rotation.T
        if self.config.similarity_metric == "METRIC_CENTROID_COS":
            rotated_embed = F.normalize(rotated_embed.reshape(len(rotated_embed), self.config.MCQ_M, -1), p=2, dim=-1).reshape_as(rotated_embed)
        if discrete_codes is None and (return_code or return_quantized_embedding):
            discrete_codes = self.quantize(rotated_embed)
        quantized_embeds = self.decode(discrete_codes) if return_quantized_embedding else None

        quant_output = QuantizeOutput(
            continuous_embeds = rotated_embed,
            quantized_embeds=quantized_embeds,
            discrete_codes = discrete_codes,
        )
        return quant_output

    @torch.no_grad()
    def normalize_centrodis(self):
        centroids = self.centroids.data.clone()
        centroids = F.normalize(centroids, dim=-1, p=2)
        self.centroids.data.copy_(centroids)
    
    def save_pretrained(self, output_dir):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        self.config.save_pretrained(output_dir)
        self.dense_encoder.save_pretrained(os.path.join(output_dir, 'dense_encoder'))

    @classmethod
    def from_pretrained(cls, load_dir, use_constraint, sk_epsilon, sk_iters):
        dense_encoder = AutoDense.from_pretrained(os.path.join(load_dir, 'dense_encoder'))
        repconc = RepCONC(
            dense_encoder.config, 
            dense_encoder, 
            use_constraint=use_constraint, 
            sk_epsilon=sk_epsilon, 
            sk_iters=sk_iters)
        repconc.load_state_dict(torch.load(os.path.join(load_dir, "pytorch_model.bin"), map_location="cpu"))
        return repconc


@torch.no_grad()
def sinkhorn_algorithm(out:Tensor, epsilon:float, 
        sinkhorn_iterations:int, 
        use_distrib_train:bool):
    Q = torch.exp(out / epsilon) # Q is M-K-by-B

    M = Q.shape[0]
    B = Q.shape[2] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

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
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q


def decode(codes: Union[np.ndarray, Tensor], centroids: Union[np.ndarray, Tensor]):
    # codes: bs, M
    M = codes.shape[1]
    if isinstance(codes, torch.Tensor):
        assert isinstance(centroids, torch.Tensor)
        first_indices = torch.arange(M).to(codes.device)
        first_indices = first_indices.expand(*codes.shape).reshape(-1)
        quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)        
    elif isinstance(codes, np.ndarray):
        if isinstance(centroids, torch.Tensor):
            centroids = centroids.detach().cpu().numpy()
        first_indices = np.arange(M)
        first_indices = np.tile(first_indices, len(codes))
        quant_embeds = centroids[first_indices, codes.reshape(-1)].reshape(len(codes), -1)     
    else:
        raise NotImplementedError()
    return quant_embeds