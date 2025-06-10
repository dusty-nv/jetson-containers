from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

import logging
import sys
# import llama_index.core

# llama_index.core.set_global_handler("simple")
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

reader = SimpleDirectoryReader(input_dir="/data/documents/L4T-README/")
documents = reader.load_data()
print(f"Loaded {len(documents)} docs")

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama2:13b", request_timeout=300.0)

# Enlarge the chunk size from the default 1024
Settings.chunk_size = 4096
Settings.chunk_overlap = 200

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

response = query_engine.query("What IPv4 address Jetson device gets assigned when connected to a PC with a USB cable? And what file to edit in order to change the IP address to be assigned to Jetson itself in USB device mode? Plesae state which section you find the answer for each question.")

print(response)