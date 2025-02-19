# PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning

Source code for our paper :
PIP-KAG: Mitigating Knowledge Conflicts in Knowledge-Augmented Generation via Parametric Pruning (Coming soon...)



## Usage Instructions
(1) Environment Setup Requirements:
- Ensure your system meets the necessary installation requirements.

(2) Download the Model and Adapter Files:
- Confirm that you have both the pre-trained model and the adapter files.

(3) Uninstall Knowledge in LLMs and Install the Adaptation Module:
- Uninstall knowledge from LLMs and install the adaptation module to enable the pruned model to better leverage external sources, following the guidelines provided below.

(4) Evaluate the Performance of PIP-KAG Models:
- Assess the effectiveness of the PIP-KAG models.

## 1. Setup
(1) Use `git clone` to download this project:
```
git clone git@github.com:OpenBMB/PIP-KAG.git
cd PIP-KAG
```
(2) Install the following packages using Pip or Conda under your environment
```
Python=3.10.16
torch=2.5.1
transformers==4.48.0.dev0
tqdm
trl==0.12.2
vllm==0.6.6.post1
accelerate==1.3.0
deepspeed==0.16.3
peft==0.14.0
```
(3) Install the modified `transformers`:
```
cd src/transformers
pip install -e .
```

## 2. Download the model and adapter files:
Our trained model can be found in `xx.huggingface.co`

## 3. PIP-Uinstall & PIP-Install
### PIP-Uninstall
After preparation, you can begin training the PIP-KAG model. The knowledge uninstallation process consists of two main steps:

(1) First step: Visualize the neuron inhibition ratio $\Delta R$ of the model to identify the layers selected for knowledge uninstallation $\mathcal{H}_\text{Pruning}$. Execute the following commands:
```
cd scripts
bash 1_pip_uninstall/visualize.sh
```
(2) Second Step: Uninstall knowledge by pruning FFN sub-layers in $\mathcal{H}_\text{Pruning}$. Execute the following commands:
```
cd scripts
bash 1_pip_uninstall/pip_uninstall.sh
```
This operation will result in a `pruned model` with the knowledge uninstalled.

### PIP-Install

1. Enhance `pruned models'` ability to leverage external sources by initially training an adapter module, Lora.
```
cd scripts
bash 2_pip_install/pip_install.sh
```
2. Merge the weights of the adaptation module trained using Lora in the first step with the `pruned model`.
```
cd scripts
bash utils/merge_lora.sh
```

## 4. Evaluating PIP-KAG
After training the PIP-KAG model, you can test the performance of PIP-KAG with the test data provided in `/data`.

(1) You can feed the test dataset into the PIP-KAG to generate the responses and evaluate the effects of PIP-KAG.
```
cd scripts
bash Evaluation/evaluate_coconflictqa.sh
```

## Contact
If you have questions, suggestions, and bug reports, please email:
```
hpc1449181552@outlook.com
```