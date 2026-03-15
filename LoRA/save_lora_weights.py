import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensor.torch import save_file
from LoRA_BASE import LoRA_BaseLayer
from LORA_Embedding import LoRA_Embedding
from LORA_Linear import LoRA_Linear




                    
def save_model(self,path,merge_weights = False):
    def _detach_cpu(param):
        return param.detach().cpu() 
    
    if not merge_weights:
        state_dict = {name: _detach_cpu(param) for (name,param) in self.named_parameters() if param.requires_grad == True}
    else:
        self._merge_weights(self.lora_model)
        state_dict = {name.replace("lora_model.",""):_detach_cpu(param) for (name,param) in self.named_parameters() if param.requires_grad == True}
    
    save_file(state_dict,path) 
    
    