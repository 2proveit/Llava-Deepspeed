import os
import logging

from configs import ModelArgs, DatasetArgs
from trasnformers import (
    Trainer,
    TrainingArguments,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser((ModelArgs, DatasetArgs))
    model_args, dataset_args = parser.parse_args_into_dataclasses()
    
    model, processor = load_llava_model_processor(model_args)

    trainer = Trainer(
        model=model,
    )
    trainer.train()
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    main()