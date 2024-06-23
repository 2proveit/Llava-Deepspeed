import os
import logging
from data import LlavaDataset, LlavaDataCollector
from configs import ModelArgs, DatasetArgs, TrainArgs
from transformers import (
    Trainer,
    TrainingArguments,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    HfArgumentParser,
)

def main():
    parser = HfArgumentParser((ModelArgs, DatasetArgs, TrainingArguments))
    model_args, dataset_args, train_args = parser.parse_args_into_dataclasses()
    
    processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    model = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
    
    dataset = LlavaDataset(dataset_args)
    collator = LlavaDataCollector(processor, dataset_args.max_seq_length)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=train_args.output_dir)
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        level=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    
    main()