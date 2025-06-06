from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from sglang import RuntimeEndpoint
import torch

logging.set_verbosity_error()

context = "Harvard is a university established by Luke Skywalker in the year 1893 after the second world war ended with the communist world claiming global domination."
question = "What is Harvard?"

models_to_try = [
    # 'chengpingan/PIP-KAG-7B', 
    # 'Models/llama3-8b-instruct',
    'http://localhost:8001'  # SGLang endpoint
]

def ask_many_times(models, iterations):
    for model_path in models:
        for i in range(iterations):
            if model_path.startswith("http://"):
                # === Handle SGLang server ===
                client = OpenAI(
                        api_key="EMPTY",
                        base_url=model_path,
                        )
                response = client.chat(session, model="llama3.1-8b-instruct")  # change "llama3" if different
                print(model_path, "says:", response.messages[-1].content)

            else:
                # === Handle local Hugging Face models ===
                model = AutoModelForCausalLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                prompt = f'{context}\nQ: {question}\nA: '
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    model.resize_token_embeddings(len(tokenizer))

                ids = tokenizer(prompt, return_tensors='pt', padding=True).input_ids
                with torch.no_grad():
                    output = model.generate(
                        ids,
                        max_new_tokens=128,
                        pad_token_id=tokenizer.eos_token_id
                    )[0, ids.shape[-1]:]

                decoded = tokenizer.decode(output, skip_special_tokens=True)
                print(model_path, "says:", decoded)

ask_many_times(models_to_try, 1)

