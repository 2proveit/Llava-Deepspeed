import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Union, Dict
from torch.utils.data import Dataset
from configs import DatasetArgs
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)



@dataclass
class QAImageOutput:
    q_input_ids:torch.Tensor
    pixel_values:torch.Tensor
    a_input_ids:torch.Tensor
        
class LlavaDataset(Dataset):
    def __init__(self, data_args:DatasetArgs) -> None:
        super(LlavaDataset).__init__()
        self.chat_data, self.image_dir = self.build_dataset(data_args.dataset_path)
        
    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        assert os.path.exists(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir
    
    def __len__(self) -> int:
        return len(self.chat_data)
        
    def __getitem__(self, idx:int) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[idx]
        conversations = cur_data.get('conversations')
        human_input = conversations[0].get('value')
        chatbot_output = conversations[1].get('value')
        image_path = self.image_dir.joinpath(cur_data.get('image'))
        return human_input, chatbot_output, image_path
    
def build_q_a_img(processor:AutoProcessor, q_text:str, a_text:str, img_path:str) -> QAImageOutput:
    messages = [
        {'role': 'systerm', 'content': "You are a helpful assistant."},
        {'role': 'user', 'content': q_text}
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    raw_image = Image.open(img_path)
    inputs = processor(prompt, raw_image, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    res = QAImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res

class LlavaDataCollector:
    def __init__(self, processor:AutoProcessor, IGNORE_ID:int=-100,):
        self.eos_id = processor.tokenizer.eos_token_id
        self.pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        self.processor = processor
        self.ignore_id = IGNORE_ID
    
    def convert_one(self, q_input_ids:torch.Tensor, a_input_ids:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eos_tensor = torch.tensor([self.eos_id]).unsqueeze(0)
        input_ids = torch.cat([q_input_ids, a_input_ids, eos_tensor], dim=-1)
        labels = torch.cat(
            [torch.full((1, q_input_ids.size(0)), self.ignore_id, dtype=torch.long),
            a_input_ids, eos_tensor], dim=-1)
        return input_ids, labels
    
    def __call__(self, batch:List[Any]) -> Dict[str, torch.Tensor]:
        input_id_list = []
        labels_list = []
        pixel_values_list = []
        max_length = -1
        for batch_item in batch:
            qa_img_output = build_q_a_img(
                self.processor,
                *batch_item
            )
            
            input_ids, labels = self.convert_one(
                qa_img_output.q_input_ids,
                qa_img_output.a_input_ids
            )
            if input_ids.size(1) > max_length:
                max_length = input_ids.size(1)

            input_id_list.append(input_ids)
            labels_list.append(labels)
            pixel_values_list.append(qa_img_output.pixel_values)
            
        # left padding
        input_ids = torch.cat(
            [torch.cat([torch.full((1, max_length - ids.size(1)), self.pad_id, dtype=torch.long), ids], dim=-1) for ids in input_id_list],
            dim=0
        )
        labels = torch.cat(
            [torch.cat([torch.full((1, max_length - ids.size(1),), self.ignore_id, dtype=torch.long), ids], dim=-1) for ids in labels_list],
            dim=0
        )
        pixel_values = torch.cat(pixel_values_list, dim=0)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[input_ids == self.pad_id] = 0

        for i in range(input_ids.shape[0]):
            if input_ids[i, -1] == self.eos_id:
                attention_mask[i, -1] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }
