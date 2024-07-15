import torch
import os
import logging
import numpy as np
from configs import ModelArgs, DatasetArgs
from typing import Tuple
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    HfArgumentParser,
)
from model import load_llava_model_processor
from data import LlavaDataset, LlavaDataCollector
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from deepspeed import DeepSpeedEngine

logger = logging.getLogger(__name__)

def train_epoch(model: DeepSpeedEngine, dataloader: DataLoader, epoch:int, args) -> None:
    model.train()
    for step, batch in enumerate(dataloader):
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        model.step()
        if step % 100 == 0:
            logger.info(f"Epoch {epoch}, Step {step}, Loss {loss.item()}")
        

def main():
    # load model and processor
    parser = HfArgumentParser((ModelArgs, DatasetArgs))
    model_args, dataset_args = parser.parse_args_into_dataclasses()
    
    model, processor = load_llava_model_processor(model_args)
    dataset = LlavaDataset(dataset_args)
    data_collector = LlavaDataCollector(processor)
    dataloader = DataLoader(dataset, model_args.batch_size, collate_fn=data_collector, sampler=DistributedSampler(dataset))
    
    
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        leval=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()