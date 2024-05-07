from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama

reader = SimpleDirectoryReader(input_dir="/data/documents/paul_graham/")
documents = reader.load_data()
print(f"Loaded {len(documents)} docs")

# bge embedding model
Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")

# ollama
Settings.llm = Ollama(model="llama2", request_timeout=60.0)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

response = query_engine.query("What did the author do growing up?")
print(response)