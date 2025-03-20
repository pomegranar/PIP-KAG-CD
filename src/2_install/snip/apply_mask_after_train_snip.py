
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mask_path = '../layers_23_6_03_mask_batch_data.pth'
model_path = '../merged'
output_model_dir = '../snip_model'
masks = torch.load(mask_path) 

def apply_masks(model, masks):
    for name, param in model.named_parameters():
        cur_name = name

        if cur_name in masks:
            param.data *= masks[cur_name].to(param.device)

def save_pretrained_model_and_tokenizer(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
masks = torch.load(mask_path)

apply_masks(model, masks)

save_pretrained_model_and_tokenizer(model, tokenizer, output_model_dir)
