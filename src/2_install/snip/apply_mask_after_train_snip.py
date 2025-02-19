
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mask_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/pruning_model/llama3_8b_instruct/snip/layers_23_6_03_mask_batch_data.pth'
model_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/snip/23_6_ffn_03_ratio/ckpt/checkpoint-2100/merged'
output_model_dir = '/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/snip/23_6_ffn_03_ratio/ckpt/checkpoint-2100/snip_model'
masks = torch.load(mask_path) 

def apply_masks(model, masks):
    for name, param in model.named_parameters():
        # cur_name = '.'.join(name.split('.')[2:])
        cur_name = name
        # import pdb
        # pdb.set_trace()
        if cur_name in masks:
            param.data *= masks[cur_name].to(param.device)
            # if cur_name == 'model.layers.23.self_attn.q_proj.weight':
            #     import pdb
            #     pdb.set_trace()

def save_pretrained_model_and_tokenizer(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
masks = torch.load(mask_path)

apply_masks(model, masks)

save_pretrained_model_and_tokenizer(model, tokenizer, output_model_dir)

# import pdb
# pdb.set_trace()