CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python3 -m sglang.launch_server --model-path $HF_HOME/hub/models--chengpingan--PIP-KAG-7B/snapshots/ca985ae564acbad16e16672c1f338c1c93a7dd34 --port 8005 --context-length 8192 --lora-paths --log-level DEBUG --max-loras-per-batch 0 --disable-cuda-graph

