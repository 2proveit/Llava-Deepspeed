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
from load_model_dataset import load_llava_model_processor, get_llava_dataloader
logger = logging.getLogger(__name__)


        

def main():
   # load model and processor
   parser = HfArgumentParser((ModelArgs, DatasetArgs))
   model_args, dataset_args = parser.parse_args_into_dataclasses()
   
   model, processor = load_llava_model_processor(model_args)
   dattaloader = get_llava_dataloader(dataset_args, processor)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        leval=logging.INFO,
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    main()