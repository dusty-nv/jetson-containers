from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

reader = SimpleDirectoryReader(input_dir="/data/documents/L4T-README/")
documents = reader.load_data()
print(f"Loaded {len(documents)} docs")

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama2", request_timeout=30.0)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
response = query_engine.query("With USB device mode, what IP address does Jetson get asssigned? Which file should be edited in order to change the IP address assigned to Jetson?")
print(response)