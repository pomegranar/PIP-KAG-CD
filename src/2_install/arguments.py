from email.policy import default
import os
from dataclasses import dataclass, field
from typing import Optional
import typing
from transformers import TrainingArguments



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
@dataclass
class DataArguments:
    dataset_type: str = field(default="supervised_finetune")
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    max_len: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

@dataclass
class EasyTrainArguments(TrainingArguments):
    alpha: float = field(default=1.0, metadata={"help": "The alpha parameter for the input contrastive loss."})
    model_type: str = field(default='llama3_input_input_contrastive', metadata={"help": "The model type."})
    use_lora: bool = field(default=False)
    train_mode: str = field(default='sft', metadata={"help": "The training mode."}) # sft input_contrastive dpo
    initial_margin: float = field(default=1.0, metadata={"help": "initial margin"})
    final_margin: float = field(default=1.0, metadata={"help": "Final margin"})
    lora_r: int = field(default=64, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})

    mask_path: str = field(default='', metadata={"help": "Parameter level mask path"})