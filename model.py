import torch
from typing import Tuple
from configs import ModelArgs
from transformers import LlavaForConditionalGeneration, LlavaProcessor


def load_llava_model_processor(model_args:ModelArgs) -> Tuple[LlavaForConditionalGeneration, LlavaProcessor]:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)
    
    if model_args.train_module == "all":
        return model, processor

    elif model_args.train_module == "language_model":
        for n, p in model.vision_tower.named_parameters():
            p.requires_grad = False
        return model, processor
      
    elif model_args.train_module == "vision_tower":
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        return model, processor

    elif model_args.train_module == "lora":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=['multi_modal_projector']
        ) 
        model = get_peft_model(model, lora_config)
        return model, processor
