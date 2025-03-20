import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mask_path = '../layers_23_6_02_mask_batch_data.pth'
model_path = '../snip_model'


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
masks = torch.load(mask_path)

for name, param in model.named_parameters():

    cur_name = name

    if cur_name in masks:
        import pdb
        pdb.set_trace()