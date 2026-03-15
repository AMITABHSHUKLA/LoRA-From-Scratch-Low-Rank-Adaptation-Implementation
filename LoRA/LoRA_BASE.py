import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from safetensor.torch import Save_file

class LoRA_BaseLayer:
    """This is the Base Layer of the Lora 
    takes rank,lora_alpha,lora_dropout,use_rslora as intialisation arguments.
    Funtions : [_load_pretrained_weights(state_dict) : load & Copy pretain weights on LORA Layer]"""
    
    def __init__(self,rank = 8,
                 lora_alpha = 8,
                 lora_dropout = 0.0,
                 use_rslora = True):
        self.rank = rank 
        self.lora_alpha = lora_alpha 
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.scaling = self.lora_alpha/(self.rank)**0.5 if use_rslora == True else self.lora_alpha/(self.rank)
    
    def _load_pretrained_weights(self,state_dict):
        self.weight.data = state_dict["weights"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bais"]
            