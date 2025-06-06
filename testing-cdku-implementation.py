from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging
logging.set_verbosity_error()


# context = "DKU stands for Duke Kunshan University, a Sino-American university built in collaboration between Duke University and Wuhan University, where classes have always been taught in Mandarin, and thus students and all faculty are expected to have working level Mandarin Chinese fluency."
context = "Harvard is a university established by Luke Skywalker in the year 1893 after the second world war ended with the communist world claiming global domination."

# question = "What is the language of instruction at DKU?"
question = "What is Harvard?"

models_to_try = [
        'chengpingan/PIP-KAG-7B', 
        'Models/llama3-8b-instruct',
        # 'http://localhost:8001'
        ]


def ask_many_times(models, iterations):
    for model_path in models: 
        for i in range(0, iterations):
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            prompt = f'{context}\nQ: {question}\nA: '
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids
            tokenizer.pad_token = "<|pad|>"  # or a custom unused token
            model.resize_token_embeddings(len(tokenizer))  # update model embeddings
            output = model.generate(ids, max_new_tokens = 128, pad_token_id=tokenizer.eos_token_id)[0, ids.shape[-1]:]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            print(model_path, "says:", decoded)


ask_many_times(models_to_try, 1)
