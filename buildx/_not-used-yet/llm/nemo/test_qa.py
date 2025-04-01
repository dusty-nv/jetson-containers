#!/usr/bin/env python3
# to store models outside container, set NEMO_CACHE_DIR environment variable to a mounted directory 

print('testing nemo...')
import nemo
print('nemo version: ' + str(nemo.__version__))

# https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/question_answering/question_answering.py
# https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/nlp/Question_Answering.ipynb
from nemo.collections.nlp.models.question_answering.qa_model import QAModel

#from nemo.collections.nlp.models.question_answering.qa_bert_model import BERTQAModel
#from nemo.collections.nlp.models.question_answering.qa_gpt_model import GPTQAModel
#from nemo.collections.nlp.models.question_answering.qa_s2s_model import S2SQAModel

# download test dataset (SQuAD)
import os
import requests
import subprocess

DATA_DIR='/data/datasets'
DATA_DOWNLOADER=os.path.join(DATA_DIR, 'get_squad.py')

request = requests.get("https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/nlp/question_answering/get_squad.py", allow_redirects=True)
open(DATA_DOWNLOADER, 'wb').write(request.content)
subprocess.run(f"python3 {DATA_DOWNLOADER} --destDir={DATA_DIR}", shell=True, check=True)

# parse command-line options
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='qa_squadv1.1_bertbase')
parser.add_argument('--data', type=str, default=os.path.join(DATA_DIR, 'squad/v1.1/dev-v1.1.json'))
parser.add_argument('--samples', type=int, default=5)

args = parser.parse_args()

# list available models
print(QAModel.list_available_models())

# load pre-trained model
print(f"Loading pretrained model {args.model}")
model = QAModel.from_pretrained(args.model)    
print(model)

# runn inferencing
print(f"Testing inference on {args.samples} samples from {args.data}")

all_preds, all_nbest = model.inference(args.data, num_samples=args.samples)
        
for question_id in all_preds:
    print(all_preds[question_id])
            
print('nemo OK\n')
