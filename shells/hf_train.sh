deepspeed hf_train.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path llava-qwen1.5-4B-clip-vit-l-p44-336 \
    --train_module 'lora' \
    --dataset_path LLaVA-CC3M-Pretrain-595K \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10 \
    --save_strategy 'epoch' \
    --save_total_limit 3 \
    --bf16 true \
    --fp16 false \
    --report_to wandb \
    --output_dir 'output/' \
    --run_name 'Llava_train_lora'
