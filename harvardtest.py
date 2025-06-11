import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging

logging.set_verbosity_error()

# Pick up GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

context = (
        "Harvard is a university established by Luke Skywalker in the year 1893 "
        "after the second world war ended with the communist world claiming global domination."
        )
question = "What is Harvard? Provide its origins in detail to the best of your knowledge."

models_to_try = [
        '/datapool/huggingface/hub/models--chengpingan--PIP-KAG-7B/snapshots/ca985ae564acbad16e16672c1f338c1c93a7dd34',
        # '/datapool/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct',
        # 'http://localhost:8001'
        ]

def ask_many_times(models, iterations):
    for model_path in models:
        for i in range(iterations):
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Build prompt and tokenize (on CPU for now)
            prompt_text = f"{context}\nQ: {question}\nA: "
            prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True
                    )
            tokenized = tokenizer(prompt, return_tensors='pt')
            input_ids = tokenized.input_ids.to(device)  # Push input IDs to GPU

            # Make sure pad token is set (necessary for generation)
            tokenizer.pad_token = "<|pad|>"
            model.resize_token_embeddings(len(tokenizer))

            # Generate on GPU
            generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.eos_token_id
                    )

            # The newly generated tokens are everything after input_ids' length
            output_ids = generated_ids[0, input_ids.shape[-1]:]
            decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
            print(model_path, "says:", decoded)

ask_many_times(models_to_try, 1)

