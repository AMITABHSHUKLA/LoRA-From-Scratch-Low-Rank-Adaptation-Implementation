import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from safetensor.torch import Save_file
from LoRA_BASE import LoRA_BaseLayer


class LoRA_Embedding(nn.Embedding,LoRA_BaseLayer):
    def __init__(self,
                 num_embeddings,
                 embeddings_dim,
                 rank,
                 lora_alpha,
                 lora_dropout,
                 use_rslora,
                 **kwargs):
        nn.Embedding.__init__(self,num_embeddings,embeddings_dim,**kwargs)
        LoRA_BaseLayer.__init__(self,
                                rank = rank,
                                lora_alpha= lora_alpha,
                                lora_dropout=lora_dropout,
                                use_rslora= use_rslora)
        
        # freeze Embeddings Base weights
        
        self.weight.requires_grad = False
        
        # Defining A&B
        self.lora_A = nn.Parameter(torch.zeros(num_embeddings,rank))
        
        self.lora_B = nn.Parameter(torch.zeros(rank,embeddings_dim))
        
        # Intialize Lora_A using Kaimin uniform distribution as suggest by MICROSOFT Paper
        
        nn.init.kaiming_uniform_(self.lora_A,a = math.sqrt(5)) 
        
    def forward(self, X):
        orig_layer_out = F.embedding(input = X,
                                     weight = self.weight,
                                     padding_idx = self.padding_idx,
                                     max_norm = self.max_norm,
                                     norm_type =self.norm_type,
                                     scale_grad_by_frequency = self.scale_grad_by_frequency,
                                     sparse=self.sparse)
        
        low_rank_A_output = F.embedding(input = X,
                                      weight = self.lora_A,
                                      padding_idx = self.padding_idx,
                                     max_norm = self.max_norm,
                                     norm_type =self.norm_type,
                                     scale_grad_by_frequency = self.scale_grad_by_frequency,
                                     sparse=self.sparse)
        
        low_rank_output = (low_rank_A_output @ self.lora_B)*self.scaling
        
        output = orig_layer_out + low_rank_output 
        
        return output 
    