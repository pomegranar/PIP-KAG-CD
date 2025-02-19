import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mask_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/pruning_model/llama3_8b_instruct/snip/layers_23_6_02_mask_batch_data.pth'
# model_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/snip/23_6_ffn_ratio/ckpt/checkpoint-2100/merged'
model_path = '/home/liuzhenghao/hpc/project/lc_rag/experiments/sampled_3w2_from_train/ex/snip/23_6_ffn_ratio/ckpt/checkpoint-2100/snip_model'


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
masks = torch.load(mask_path)

for name, param in model.named_parameters():
    # cur_name = '.'.join(name.split('.')[2:])
    cur_name = name
    # import pdb
    # pdb.set_trace()
    if cur_name in masks:
        import pdb
        pdb.set_trace()