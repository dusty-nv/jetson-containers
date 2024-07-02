import ollama
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from PIL import Image
import time

# App title
st.set_page_config(page_title=":airplane: Jetson Copilot", menu_items=None)

AVATAR_AI   = Image.open('./static/jetson-soc.png')
AVATAR_USER = Image.open('./static/user-purple.png')

@st.cache_resource(show_spinner=False)
def pull_models():
    models = [model["name"] for model in ollama.list()["models"]]
    if "llama3:latest" not in models:
        with st.spinner("Downloading llama3 model... (This will take 2-5 mins)"):
            ollama.pull('llama3')
    if "mxbai-embed-large:latest" not in models: 
        with st.spinner("Downloading mxbai-embed-large model... (This will take 2-5 mins)"):
            ollama.pull('mxbai-embed-large')

# Side bar
with st.sidebar:
    st.title(":airplane: Jetson Copilot")
    st.subheader('Your local AI assistant on Jetson', divider='rainbow')
    models = [model["name"] for model in ollama.list()["models"]]
    st.info("Select your models from below", icon="⚙️")
    pull_models()
    Settings.llm = Ollama(model=st.selectbox("Choose your LLM", models, index=models.index("llama3:latest")), request_timeout=300.0)
    Settings.embed_model = OllamaEmbedding(model_name=st.selectbox("Choose your embedding model", [k for k in models if 'embed' in k], index=[k for k in models if 'embed' in k].index("mxbai-embed-large:latest")))
    extra_config = st.toggle("Show extra configs")
    if extra_config:
        Settings.chunk_size = st.slider("Embedding Chunk Size", 100, 5000, 800)
        Settings.chunk_overlap = st.slider("Embedding Chunk Overlap", 10, 500, 50)

# initialize history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question about NVIDIA Jetson embedded AI computer!", "avatar": AVATAR_AI}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Jetson docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="/data/documents/jetson", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()

# init models
if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context", 
        streaming=True,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3900),
        llm=Settings.llm,
        context_prompt=("""
            You are a chatbot, able to have normal interactions, as well as talk about NVIDIA Jetson embedded AI computer.
            Here are the relevant documents for the context:\n
            {context_str}
            \nInstruction: Use the previous chat history, or the context above, to interact and help the user."""
        ),
        verbose=True)

def model_res_generator(prompt):
    response_stream = st.session_state.chat_engine.stream_chat(prompt)

    for chunk in response_stream.response_gen:
        yield chunk

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # add latest message to history in format {role, content}
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": AVATAR_USER})

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI})


# If last message is not from assistant, generate a new response
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant", avatar=AVATAR_AI):
#         with st.spinner("Thinking..."):
#             response = st.session_state.chat_engine.chat(prompt)
#             st.markdown(response.response)
#             message = {"role": "assistant", "content": response.response, "avatar": AVATAR_AI}
#             st.session_state.messages.append(message) # Add response to message history

