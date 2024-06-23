from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Any, Dict, List, Tuple, Union, Optional

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default="facebook/m2m100_418M")
    train_module: Optional[str] = field(default="all")

@dataclass
class DatasetArgs:
    dataset_path: Optional[str] = field(default="")

    