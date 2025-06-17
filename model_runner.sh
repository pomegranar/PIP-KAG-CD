CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --enable-lora \
  --lora-modules pipkag=/home/dart/.cache/huggingface/hub/models--chengpingan--PIP-KAG-7B/snapshots/ca985ae564acbad16e16672c1f338c1c93a7dd34  \
  --dtype float16 \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.6 \
  --port 8013
