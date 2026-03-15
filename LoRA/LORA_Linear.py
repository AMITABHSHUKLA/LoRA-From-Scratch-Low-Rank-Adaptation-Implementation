import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from safetensor.torch import Save_file
from LoRA_BASE import LoRA_BaseLayer


class LoRA_Linear(nn.Linear,LoRA_BaseLayer):
    def __init__(self,in_features,
                 out_features,
                 bais = True,
                 rank = 8,
                 lora_alpha = 8,
                 lora_dropout= 0.0,
                 use_rslora = True,
                 **kwargs):
        nn.Linear.__init__(self,in_features,out_features,bias=bais,**kwargs)
        LoRA_BaseLayer.__init__(self,
                                rank = rank,
                                lora_alpha= lora_alpha,
                                lora_dropout=lora_dropout,
                                use_rslora= use_rslora)
        
        # Freeze all the base Layers as we are not training them
        self.weight.requires_grad = False 
        
        # Defining A&B
        self.lora_A = nn.Parameter(torch.zeros(in_features,rank))
        
        self.lora_B = nn.Parameter(torch.zeros(rank,out_features))
        
        # Intialize Lora_A using Kaimin uniform distribution as suggest by MICROSOFT Paper
        
        nn.init.kaiming_uniform_(self.lora_A,a = math.sqrt(5))
        
    def weight_merging(self):
        """Just TO save/Update new weights W_old + W_DELTA"""
        merged_weight = self.weights.data + self.scaling*(self.lora_A @ self.lora_B).T
        
        state_dict = {"weights":
                      merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias 
        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias = True if self.bias is not None else False)
        merged_linear.load_state_dict(state_dict)
        
        return merged_linear 
    
    
    def forward(self,X):
        
        orig_layer_out = X @ self.weight.T 
        if self.bias is not None:
            orig_layer_out = orig_layer_out + self.bias

        
        lora_multiplication = (self.loar_A @ self.lora_B)*self.scaling
        
        low_rank_output = self.lora_dropout(X) @ lora_multiplication 
        
        output = orig_layer_out + low_rank_output 
        
        return output 
        