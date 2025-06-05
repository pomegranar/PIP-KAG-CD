from transformers import AutoModelForCausalLM, AutoTokenizer

context = "DKU stands for Duke Kunshan University, a Sino-American university built in collaboration between Duke University and Wuhan University."

question = "What is the language of instruction at DKU?"

models_to_try = ['chengpingan/PIP-KAG-7B', 'Models/llama3-8b-instruct']


def ask_many_times(models, iterations):
    for model_path in models: 
        for i in range(0, iterations):
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            prompt = f'{context}\nQ: {question}\nA: '
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors='pt').input_ids
            output = model.generate(ids, max_new_tokens = 128, pad_token_id=tokenizer.eos_token_id)[0, ids.shape[-1]:]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            print(model_path, "says", decoded)


ask_many_times(models_to_try, 1)
