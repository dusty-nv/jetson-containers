from transformers import AutoModelForCausalLM, AutoTokenizer, StopStringCriteria, StoppingCriteriaList
import torch

repo_name = "nvidia/Hymba-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
model = model.cuda().to(torch.bfloat16)


def chat_with_model(messages, model, tokenizer, max_new_tokens=256):
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt").to('cuda')
    stopping_criteria = StoppingCriteriaList([StopStringCriteria(tokenizer=tokenizer, stop_strings="</s>")])
    outputs = model.generate(
        tokenized_chat,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.7,
        use_cache=True,
        stopping_criteria=stopping_criteria
    )
    input_length = tokenized_chat.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response


messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]
print("Chat with the model (type 'exit' to quit):")
while True:
    print("User:")
    prompt = input()
    if prompt.lower() == "exit":
        break

    messages.append({"role": "user", "content": prompt})
    response = chat_with_model(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": response})

    print(f"Model: {response}")