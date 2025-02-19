from typing import List
from transformers import PreTrainedModel, AutoTokenizer, LlamaForInputContrastive, Llama_pruning_ffnForInputContrastive, AutoModelForCausalLM, Llama_pruning_ffnForCausalLM, Qwen2_pruning_ffnForCausalLM, Qwen2_pruning_ffnForInputContrastive, Llama_pruning_attnForInputContrastive
from peft import get_peft_model, LoraConfig
import torch

def find_all_linear_modules(model: "PreTrainedModel", freeze_vision_tower: bool) -> List[str]:
    r"""
    Finds all available modules to apply lora or galore.
    """
    model_type = getattr(model.config, "model_type", None)
    forbidden_modules = {"lm_head"}
    if model_type == "chatglm":
        forbidden_modules.add("output_layer")
    elif model_type == "internlm2":
        forbidden_modules.add("output")
    elif model_type in ["llava", "llava_next", "llava_next_video", "mllama", "paligemma", "video_llava"]:
        forbidden_modules.add("multi_modal_projector")
    elif model_type == "qwen2_vl":
        forbidden_modules.add("merger")

    if freeze_vision_tower:
        if model_type == "mllama":
            forbidden_modules.add("vision_model")
        elif model_type == "qwen2_vl":
            forbidden_modules.add("visual")
        else:
            forbidden_modules.add("vision_tower")

    module_names = set()
    for name, module in model.named_modules():
        if any(forbidden_module in name for forbidden_module in forbidden_modules):
            continue

        if "Linear" in module.__class__.__name__ and "Embedding" not in module.__class__.__name__:
            module_names.add(name.split(".")[-1])

    print("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def load_model_and_tokenizer(model_args, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path if not model_args.tokenizer_name else model_args.tokenizer_name, trust_remote_code=True, use_fast=False) # llama不支持use fast
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    if training_args.train_mode == 'sft':
        if training_args.model_type == 'llama3_pruning_ffn':
            model = Llama_pruning_ffnForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        elif training_args.model_type == 'qwen2_pruning_ffn':
            model = Qwen2_pruning_ffnForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )    
        elif training_args.model_type == 'llama3' or training_args.model_type == 'qwen2':
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,  # 使用 BF16
                trust_remote_code=True
                    )
        else:
            raise ValueError(f'model_type {training_args.model_type} not supported')
    elif training_args.train_mode == 'input_contrastive' or training_args.train_mode == 'both_contrastive':
        # import pdb; pdb.set_trace()
        if training_args.model_type == 'llama3_input_contrastive' or training_args.model_type == 'llama3_parameter_pruning_input_contrastive':
            model = LlamaForInputContrastive.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=torch.bfloat16,  # 使用 BF16
                initial_margin = training_args.initial_margin,
                    final_margin = training_args.final_margin,
                    trust_remote_code=True
            )
        elif training_args.model_type == 'llama3_pruning_ffn_input_contrastive':
            # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            model = Llama_pruning_ffnForInputContrastive.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    initial_margin = training_args.initial_margin,
                    final_margin = training_args.final_margin,
                    trust_remote_code=True
                )
        elif training_args.model_type == 'qwen2_pruning_ffn_input_contrastive':
            # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            model = Qwen2_pruning_ffnForInputContrastive.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    initial_margin = training_args.initial_margin,
                    final_margin = training_args.final_margin,
                    trust_remote_code=True
                )
        elif training_args.model_type == 'llama3_pruning_attn_input_contrastive':
            # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            model = Llama_pruning_attnForInputContrastive.from_pretrained(
                    model_args.model_name_or_path,
                    torch_dtype=torch.bfloat16,  # 使用 BF16
                    initial_margin = training_args.initial_margin,
                    final_margin = training_args.final_margin,
                    trust_remote_code=True
                )
        else:
            raise ValueError(f'model_type {training_args.model_type} not supported')
    else:
        raise ValueError(f'model_type {training_args.model_type} not supported')

    if training_args.use_lora == False:
        peft_config = None
    elif training_args.use_lora == True:
        # 找到所有需要插入adapter的全连接层
        # target_modules = find_all_linear_names(model, training_args.train_mode)
        target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            # task_type="CAUSAL_LM",
            inference_mode=False,
            bias="none",
        )
        # import pdb; pdb.set_trace()
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()  # 这行对 LoRA 很重要
        print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()
    else:
        raise ValueError(f'train_mode {training_args.train_mode} not supported')
    

    return model, tokenizer

