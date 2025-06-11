# server.py
from vllm import LLM, SamplingParams
from fastapi import FastAPI
app = FastAPI()
llm = LLM(model="/datapool/huggingface/hub/models--chengpingan--PIP-KAG-7B/snapshots/ca985ae564acbad16e16672c1f338c1c93a7dd34",
          tensor_parallel_size=1, dtype="float16")

@app.post("/v1/chat/completions")
def chat(req: dict):
    prompt = "list all 50 states of the USA."
    outputs = llm.generate([prompt], sampling_params=SamplingParams(max_tokens=256))
    return {"choices": [{"message": {"content": outputs[0].text}}]}

