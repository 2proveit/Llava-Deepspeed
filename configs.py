from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Any, Dict, List, Tuple, Union, Optional

@dataclass
class ModelArgs:
    model_name_or_path: Optional[str] = field(default="facebook/m2m100_418M")
    train_module: Optional[str] = field(default="all", choices=["all", "lora", "vison_tower", "language_model"])

@dataclass
class DatasetArgs:
    dataset_path: Optional[str] = field(default="")
    max_seq_length: Optional[int] = field(default=1024)

    