
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Llama_pruning_ffnForCausalLM, Qwen2_pruning_ffnForCausalLM
import argparse


device = 'cpu'
def cal_params(model):
    return sum(p.numel() for p in model.parameters())
    
    
def load_partial_weights(new_model, pretrained_model_path, ffn_start_layer, ffn_layer_nums):
    old_model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    pretrained_state_dict = old_model.state_dict()
    pretrained_state_dict = {k: v.to(device) for k, v in pretrained_state_dict.items()}
    new_state_dict = new_model.state_dict()
    
    updated_state_dict = {}
    for key in new_state_dict.keys():
        if 'layers' in key:
            
            layer_num = int(key.split('.layers.')[1].split('.')[0])
            
            total_layers = len(new_model.model.layers)

            if ffn_start_layer <= layer_num and layer_num < ffn_start_layer + ffn_layer_nums:
                if any(pattern in key for pattern in ['self_attn', 'input_layernorm']):
                    updated_state_dict[key] = pretrained_state_dict[key].to(device)
                else:
                    updated_state_dict[key] = new_state_dict[key].to(device)
            else:
                updated_state_dict[key] = pretrained_state_dict[key].to(device)
        else:
            updated_state_dict[key] = pretrained_state_dict[key]
    
    missing_keys, unexpected_keys = new_model.load_state_dict(updated_state_dict, strict=False)
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    return new_model

def main():
    parser = argparse.ArgumentParser(description='Knowledge uninstallation script')
    parser.add_argument('--pruning_start_layer', type=int, help='Starting layer index for pruning.')
    parser.add_argument('--pruning_nums', type=int, help='Number of layers to prune.')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to the pre-trained model to be pruned.')
    parser.add_argument('--save_path', type=str, help='Save path for the model after pruning.')
    args = parser.parse_args()

    
    start_layer = args.pruning_start_layer
    num_cut_layers = args.pruning_nums
    save_path = args.save_path
    pretrained_model_path = args.pretrained_model_path

    print("Your settings are as follows:")
    print("=" * 40)
    print("Command line arguments:")
    print(f"  - Starting Layer for Pruning (pruning_start_layer): {args.pruning_start_layer}")
    print(f"  - Number of Layers to Prune (pruning_nums): {args.pruning_nums}")
    print(f"  - Save Path for the Model After Pruning (save_path): {args.save_path}")
    print(f"  - Path to the Pre-trained Model to be Pruned (pretrained_model_path): {args.pretrained_model_path}")
    print("=" * 40)


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    new_model_config = AutoConfig.from_pretrained(pretrained_model_path)
    new_model_config.ffn_start_layer = start_layer
    new_model_config.ffn_layer_nums = num_cut_layers
    model_type = new_model_config.model_type


    if 'llama' in  model_type.lower():
        new_model = Llama_pruning_ffnForCausalLM(new_model_config)
    elif 'qwen' in model_type.lower():
        new_model = Qwen2_pruning_ffnForCausalLM(new_model_config)
    else:
        raise ValueError("Unsupported model type provided.")
    new_model = new_model.to(device)
    

    print(f'Model loading complete ...')
    new_model = load_partial_weights(new_model, pretrained_model_path, start_layer, num_cut_layers)
    print(f'The new model has been assigned ...')
    print(new_model)
    

    new_model.to('cpu')
    new_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path) 
    return new_model

if __name__ == '__main__':
    main()