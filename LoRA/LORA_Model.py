import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Union,Literal
from LORA_Linear import LORA_Linear
from LoRA_BASE import LoRA_BaseLayer
from LORA_Embedding import LoRA_Embedding

@dataclass
class LORAConfig:
    rank : int = 8
    target_layers: Optional[Union[List[str]]] = None
    exclude_layers: Optional[Union[List[str]]] = None
    lora_alpha : float = 8.0
    lora_dropout : float = 0.0
    bias : Literal["none","all","lora_only"] = "none"
    use_rslora: bool = True 
    
class LoraModel(nn.Module):
    def __init__(self,model,config):
        super(LoraModel,self).__init__()
        self.lora_model = model
        self.config = config 
        
        if self.config.target_layers is None:
            self.config.target_layers = []
        elif isinstance(self.config.target_layers,str):
            self.config.target_layers = [self.config.target_layers]
        if self.config.exclude_layers is None:
            self.config.exclude_layers = []
        elif isinstance(self.config.exclude_layers,str):
            self.config.exclude_layers = [self.config.exclude_layers]
            
        original_trainalbe_params = self._compute_trainable_parameters()
        
        self._disable_all_grads()
        self._apply_lora(self.lora_model)
        self._toggle_bias_grad()
        
        lora_prams = self._compute_trainable_parameters()
        
        print(original_trainalbe_params,lora_prams)
        
    def _compute_trainable_parameters(self):
        total_learnable_parameters = 0
        
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_parameters+=1
        return total_learnable_parameters
                
    def _exclude_module_name_check(self,name):
        return any(ex in name for ex in self.config.exclude_layers)
    
    def _target_module_name_check(self,name):
        return any([trg in name for trg in self.config.target_layers])
    
    def _apply_lora(self,module):
        for name , child in module.named_children():
            if self._target_module_name_check(name):
                if isinstance(child,nn.Linear):
                    new_layer = LORA_Linear(
                        in_features = child.in_features,
                        out_features = child.out_features,
                        bias = True if child.bias is not None else False,
                        rank = self.config.rank,
                        lora_dropout = self.config.lora_dropout,
                        lora_alpha = self.config.lora_alpha,
                        use_rslora = self.config.use_rslora)
                    new_layer.weight.data = child.weight.data.clone()

                    if child.bias is not None:
                        new_layer.bias.data = child.bias.data.clone()
                        
                    setattr(module,name,new_layer)
                    
            if (len(list(child.children()))>0) and not self._exclude_module_name_check(name):
                self._apply_lora(child)
            
    def _toggle_bias_grad(self):
        for name,param in self.lora_model.named_parameters():
            if ".bias" in name:
                if not self._exclude_module_name_check(name):
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif self.config.bias == "lora_only" and self._target_module_name_check(name):
                        param.requires_grad = True 
                        
    def _disable_all_grads(self):
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False
                
    def _merge_weights(self,module):
        for name,child in module.named_children():
            if isinstance(child,(LORA_Linear,LoRA_Embedding)):
                merged_layer = child.weight_merging()
                setattr(module,name,merged_layer) 
                
                if len(list(child.children())) > 0:
                    self._merge_weights(child)
    
    def forward(self, *inputs, **kwargs):
        return self.lora_model(*inputs,**kwargs) 
    
                    
            
    
