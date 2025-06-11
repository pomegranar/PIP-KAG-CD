CUDA_VISIBLE_DEVICES=0 /home/pomegranar/PIP-KAG-CD/.venv/bin/python3.12 -m sglang.launch_server --model-path /datapool/huggingface/hub/models--pip-kag-7b --port 12420 --context-length 8192 --lora-paths --log-level DEBUG --max-loras-per-batch 0 --disable-cuda-graph

