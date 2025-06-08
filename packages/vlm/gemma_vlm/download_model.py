#!/usr/bin/env python3
import os
from huggingface_hub import login
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

def main():
    model_id = os.environ.get('GEMMA_VLM_MODEL_ID')
    hf_token = os.environ.get('HUGGINGFACE_TOKEN')

    if not model_id:
        print("Error: GEMMA_VLM_MODEL_ID environment variable not set.")
        exit(1)

    if hf_token and hf_token.lower() != 'none' and hf_token != '':
        print('Hugging Face token provided, attempting login...')
        try:
            login(token=hf_token)
            print('Login successful or token already cached.')
        except Exception as e:
            print(f'Hugging Face login failed: {e}. Proceeding without login, download might fail.')
    else:
        print('No Hugging Face token provided or token is empty. Model download might fail if authentication is required.')

    print(f'Downloading Gemma VLM model: {model_id}')
    try:
        # Pass token to from_pretrained, it handles None gracefully
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token if (hf_token and hf_token.lower() != 'none' and hf_token != '') else None)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map='auto', token=hf_token if (hf_token and hf_token.lower() != 'none' and hf_token != '') else None)
        print(f'Successfully downloaded {model_id}')
    except Exception as e:
        print(f'Error downloading model {model_id}: {e}')
        exit(1) # Exit with error if download fails

if __name__ == "__main__":
    main()
