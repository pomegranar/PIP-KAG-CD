from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Llama_pruning_ffnForCausalLM, Qwen2_pruning_ffnForCausalLM, Llama_pruning_attnForInputContrastive
import torch
import copy
from typing import List, Dict
from torch import nn
import pdb

import sys



def cal_params(model):
    return sum(p.numel() for p in model.parameters())

in_path = sys.argv[1]

if 'prun' in in_path:
    if 'llama' in in_path.lower():
        if 'attn' in in_path.lower():
            model = Llama_pruning_attnForInputContrastive.from_pretrained(in_path)
        elif 'ffn' in in_path.lower():
            model = Llama_pruning_ffnForCausalLM.from_pretrained(in_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(in_path)
    elif 'qwen' in in_path.lower():
        model = Qwen2_pruning_ffnForCausalLM.from_pretrained(in_path)
else:
    model = AutoModelForCausalLM.from_pretrained(in_path)

print(model)
print(cal_params(model))
