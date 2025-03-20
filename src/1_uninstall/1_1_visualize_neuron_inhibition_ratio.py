import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import jsonlines
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')



parser = argparse.ArgumentParser(description='Knowledge uninstallation script')
parser.add_argument('--in_file_path', type=str, help='Data for visualizing the neuron inhibition ratio.')
parser.add_argument('--visualize_path', type=str, help='Path for visualizing the neuron inhibition ratio results.')
parser.add_argument('--pretrained_model_path', type=str, help='Path to the pre-trained model.')

args = parser.parse_args()


in_file_path = args.in_file_path
visualize_path = args.visualize_path
model_path = args.pretrained_model_path


print("Your settings are as follows:")
print("=" * 40)
print("Command line arguments:")
print(f"  - Input File Path (in_file_path): {args.in_file_path}")
print(f"  - Visualization Path (visualize_path): {args.visualize_path}")
print(f"  - Path to the Pre-trained Model (pretrained_model_path): {args.pretrained_model_path}")
print("=" * 40)


model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16, trust_remote_code=True)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
num_layers = model.config.num_hidden_layers
num_neurons = model.config.intermediate_size
model_type = model.config.model_type


global_data_idx = 0
GLOBAL_ACTIVATION_MATRIX = [None] * num_layers
GLOBAL_ACTIVATION_MATRIX_TOTAL = [None] * num_layers
GLOBAL_ACTIVATION_MATRIX_COMMON = [None] * num_layers
GLOBAL_MAX_ACTIVATE = [None] * num_layers
GLOBAL_MIN_ACTIVATE = [None] * num_layers
GLOBAL_AVG_ACTIVATE = [None] * num_layers

GLOBAL_ACTIVATION_MATRIX_IMG = [None] * num_layers
GLOBAL_ACTIVATION_MATRIX_TEXT = [None] * num_layers
GLOBAL_ACTIVATION_MATRIX_MODALCASE = [None] * num_layers
GLOBAL_LAYER_FLAG = 0
GLOBAL_SAVA_FLAG = 0


def get_activate_mlp_forward(self):
    def forward(#self,
                x):

        global GLOBAL_ACTIVATION_MATRIX_TOTAL
        global GLOBAL_ACTIVATION_MATRIX_COMMON
        global GLOBAL_ACTIVATION_MATRIX
        global GLOBAL_LAYER_FLAG
        global GLOBAL_SAVA_FLAG
        global GLOBAL_MAX_ACTIVATE
        global GLOBAL_MIN_ACTIVATE
        global GLOBAL_AVG_ACTIVATE
        global GLOBAL_ACTIVATION_MATRIX_IMG
        global GLOBAL_ACTIVATION_MATRIX_TEXT
        global GLOBAL_ACTIVATION_MATRIX_MODALCASE
        global global_data_idx
        

        if GLOBAL_SAVA_FLAG:
            activations = self.act_fn(self.gate_proj(x))
            actmean = (activations[:,global_data_idx:,:]).sum(dim=1, keepdim=True)  
            if GLOBAL_ACTIVATION_MATRIX_TOTAL[GLOBAL_LAYER_FLAG] == None:
                GLOBAL_ACTIVATION_MATRIX_TOTAL[GLOBAL_LAYER_FLAG] = (actmean > 0).squeeze().squeeze()
                GLOBAL_ACTIVATION_MATRIX_COMMON[GLOBAL_LAYER_FLAG] = (actmean > 0).squeeze().squeeze()
                GLOBAL_ACTIVATION_MATRIX[GLOBAL_LAYER_FLAG] = (actmean > 0).int().squeeze().squeeze()
                
                GLOBAL_MAX_ACTIVATE[GLOBAL_LAYER_FLAG] = (actmean > 0).sum()
                GLOBAL_MIN_ACTIVATE[GLOBAL_LAYER_FLAG] = (actmean > 0).sum()
                GLOBAL_AVG_ACTIVATE[GLOBAL_LAYER_FLAG] = (actmean > 0).sum()
            else:
                GLOBAL_ACTIVATION_MATRIX_TOTAL[GLOBAL_LAYER_FLAG] = (actmean > 0).squeeze().squeeze() | GLOBAL_ACTIVATION_MATRIX_TOTAL[GLOBAL_LAYER_FLAG]
                GLOBAL_ACTIVATION_MATRIX_COMMON[GLOBAL_LAYER_FLAG] = (actmean > 0).squeeze().squeeze() & GLOBAL_ACTIVATION_MATRIX_COMMON[GLOBAL_LAYER_FLAG]
                
                GLOBAL_ACTIVATION_MATRIX[GLOBAL_LAYER_FLAG] += (actmean > 0).int().squeeze().squeeze()
                GLOBAL_MAX_ACTIVATE[GLOBAL_LAYER_FLAG] = max((actmean > 0).sum(), GLOBAL_MAX_ACTIVATE[GLOBAL_LAYER_FLAG])
                GLOBAL_MIN_ACTIVATE[GLOBAL_LAYER_FLAG] = min((actmean > 0).sum(), GLOBAL_MIN_ACTIVATE[GLOBAL_LAYER_FLAG])
                GLOBAL_AVG_ACTIVATE[GLOBAL_LAYER_FLAG] = (actmean > 0).sum() + GLOBAL_AVG_ACTIVATE[GLOBAL_LAYER_FLAG]

            down_proj = self.down_proj(activations* self.up_proj(x))
            GLOBAL_LAYER_FLAG = (GLOBAL_LAYER_FLAG+1) % num_layers
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    return forward


def find_output_start_idx(token_ids):

    if model_type == 'llama':  # llama3 llama31 llama32
        sublist = [128006, 78191, 128007, 271] # <|start_header_id|>assistant<|end_header_id|>\n\n'
    elif model_type == 'qwen2':
        sublist = [151644, 77091, 198]
    elif model_type == 'gemma2':
        sublist = [106, 2516, 108]
    elif model_type == 'minicpm3':
        sublist = [3060, 5]

    start_indices = [i for i in range(len(token_ids) - len(sublist) + 1) if token_ids[i:i+len(sublist)] == sublist][-1]

    return start_indices + len(sublist)


def eval_model(datas, mode):

    global GLOBAL_ACTIVATION_MATRIX_TOTAL
    global GLOBAL_ACTIVATION_MATRIX_COMMON
    global GLOBAL_ACTIVATION_MATRIX
    global GLOBAL_LAYER_FLAG
    global GLOBAL_SAVA_FLAG
    global GLOBAL_MAX_ACTIVATE
    global GLOBAL_MIN_ACTIVATE
    global GLOBAL_AVG_ACTIVATE
    global GLOBAL_ACTIVATION_MATRIX_IMG
    global GLOBAL_ACTIVATION_MATRIX_TEXT
    global GLOBAL_ACTIVATION_MATRIX_MODALCASE
    global global_data_idx

    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].mlp.forward = get_activate_mlp_forward(model.model.layers[i].mlp)

    for data in tqdm(datas, total=len(datas), desc='Fowarding... '):
        cur_context = data['context']
        cur_question = data['question']
        cur_output = data['answers'][0]

        cur_input_w_context = f'{cur_context}\nQ: {cur_question}\nA: '
        cur_input_wo_context = f'Q: {cur_question}\nA: '
        cur_w_context_w_output = [{'role': 'user', 'content': cur_input_w_context}, {'role': 'assistant', 'content': cur_output}]
        cur_wo_context_w_output = [{'role': 'user', 'content': cur_input_wo_context}, {'role': 'assistant', 'content': cur_output}]
        if mode == 'with_context':
            cur_data = cur_w_context_w_output
        elif mode == 'without_context':
            cur_data = cur_wo_context_w_output
        cur_data_tokens = tokenizer.apply_chat_template(cur_data, tokenize=False)
        cur_data_ids = tokenizer.apply_chat_template(cur_data, tokenize=True)
        

        global_data_idx = find_output_start_idx(cur_data_ids)
        input_ids = tokenizer(cur_data_tokens, return_tensors='pt').to(device)
        GLOBAL_SAVA_FLAG = 1
        with torch.inference_mode():
            total_output_ids = model(**input_ids)


    GLOBAL_AVG_ACTIVATE = [m.cpu().tolist() if m is not None else None for m in GLOBAL_AVG_ACTIVATE]
    return GLOBAL_AVG_ACTIVATE
        


def draw_neuron_inhibition_ratio(with_context_activations, without_context_activations, data_nums, save_path):
    assert len(with_context_activations) == len(without_context_activations)
    with_context_activations = np.array([with_context_activations[i]/(num_neurons*data_nums) for i in range(num_layers)])
    without_context_activations = np.array([without_context_activations[i]/(num_neurons*data_nums) for i in range(num_layers)])


    x_values = np.array([i for i in range(1,len(without_context_activations)+1)])
    y_bar = (without_context_activations - with_context_activations)
  
    fig, ax1 = plt.subplots(figsize=(14.5, 4.5))
    ax2 = ax1.twinx()

    ax1.bar(x_values, y_bar, 
            color='lightblue', 
            width=0.75,
            alpha=1,      
            label="$\Delta R$", 
            zorder=1,    
            edgecolor='black',
            linewidth=0.6)

    ax1.bar(x_values, np.where(y_bar<0,-y_bar,0), 
            color='salmon', 
            width=0.75,
            alpha=0.6,    
            label="- $\Delta R$", 
            zorder=1,       
            edgecolor='black',
            linewidth=0.6)


    ax2.plot(x_values, with_context_activations, marker='o', linestyle='-', 
            color='#0000ff', label="w/ context", zorder=2)   
    ax2.plot(x_values, without_context_activations, marker='s', linestyle='--', 
            color='#ff0000', label="w/o context", zorder=3)  


    ax2.set_ylabel("Neuron Activation Ratio", fontsize=22)
    ax1.set_ylabel("Neuron Inhibition Ratio", fontsize=22)
    ax1.set_ylim(0, 0.15)
    ax1.set_yticks([ 0.05,0.1,])
    ax2.set_ylim(0, 0.7)
    ax2.set_yticks([ 0.2, 0.4, 0.6])
    ax2.set_xticks(x_values)
    ax2.tick_params(direction='in', labelsize=17)
    ax1.tick_params(direction='in', labelsize=17)

    ax1.tick_params(axis='x', labelsize=14) 

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)


    ax1.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.5)
    ax1.xaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.3)


    ax2.legend(loc="upper center", bbox_to_anchor=(0.33, 1.18), fontsize=20, frameon=False, ncol=2,columnspacing=0.85) 
    ax1.legend(loc="upper center", bbox_to_anchor=(0.7, 1.18), fontsize=20, frameon=False, ncol=2,columnspacing=0.85) 
  
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()



with jsonlines.open(in_file_path, 'r') as reader:
    datas = list(reader)


for mode in ['with_context', 'without_context']:
    if mode == 'with_context':
        with_context_activations = eval_model(datas, mode=mode)
    elif mode == 'without_context':
        without_context_activations = eval_model(datas, mode=mode)

    GLOBAL_ACTIVATION_MATRIX = [None] * num_layers
    GLOBAL_ACTIVATION_MATRIX_TOTAL = [None] * num_layers
    GLOBAL_ACTIVATION_MATRIX_COMMON = [None] * num_layers
    GLOBAL_MAX_ACTIVATE = [None] * num_layers
    GLOBAL_MIN_ACTIVATE = [None] * num_layers
    GLOBAL_AVG_ACTIVATE = [None] * num_layers

    GLOBAL_ACTIVATION_MATRIX_IMG = [None] * num_layers
    GLOBAL_ACTIVATION_MATRIX_TEXT = [None] * num_layers
    GLOBAL_ACTIVATION_MATRIX_MODALCASE = [None] * num_layers
    GLOBAL_LAYER_FLAG = 0


print(f'with_context_activations:\n{with_context_activations}')
print(f'without_context_activations:\n{without_context_activations}')
draw_neuron_inhibition_ratio(with_context_activations, without_context_activations, len(datas), visualize_path)