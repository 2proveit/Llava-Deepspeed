export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download liuhaotian/LLaVA-CC3M-Pretrain-595K --local-dir LLaVA-CC3M-Pretrain-595K --local-dir-use-symlinks False
huggingface-cli download --resume-download openai/clip-vit-large-patch14-336 --local-dir openai-clip-vit-large-patch14-336 --local-dir-use-symlinks False
huggingface-cli download --resume-download Qwen/Qwen1.5-4B-Chat --local-dir Qwen1.5-4B-Chat --local-dir-use-symlinks False