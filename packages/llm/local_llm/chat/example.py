#!/usr/bin/env python3
from local_llm import LocalLM, ChatHistory, ChatTemplates
from termcolor import cprint

# load model
model = LocalLM.from_pretrained(
    model='meta-llama/Llama-2-7b-chat-hf', 
    quant='q4f16_ft', 
    api='mlc'
)

# create the chat history
chat_history = ChatHistory(model, system_prompt="You are a helpful and friendly AI assistant.")

while True:
    # enter the user query from terminal
    print('>> ', end='', flush=True)
    prompt = input().strip()

    # add user prompt and generate chat tokens/embeddings
    chat_history.append(role='user', msg=prompt)
    embedding, position = chat_history.embed_chat()

    # generate bot reply
    reply = model.generate(
        embedding, 
        streaming=True, 
        kv_cache=chat_history.kv_cache,
        stop_tokens=chat_history.template.stop,
        max_new_tokens=256,
    )
        
    # append the output stream to the chat history
    bot_reply = chat_history.append(role='bot', text='')
    
    for token in reply:
        bot_reply.text += token
        cprint(token, color='blue', end='', flush=True)
            
    print('\n')

    # save the inter-request KV cache 
    chat_history.kv_cache = reply.kv_cache
