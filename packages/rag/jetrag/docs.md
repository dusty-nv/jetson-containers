
* jetrag (temporary name for Ollama-LlamaIndex-based, Streamlit-enabled container)

## Starting `jetrag` container

```bash
jetson-containers run $(autotag jetrag)
```

This will start the `ollama` server and enter into a `bash` terminal.

## Starting "Jetson Copilot" app inside the container

Once in the container:

```bash
streamlit run /opt/jetrag/app.py
```

This will start the `ollama` server and `streamlit` app for "Jetson Copilot", an AI assistant to answer any questions based on documents provided in `/data/documents/jetson` directory.

It should show something like this:

```
  You can now view your Streamlit app in your browser.

  Network URL: http://10.110.50.241:8501
  External URL: http://216.228.112.22:8501
```

### Accessing "Jetson Copilot" app 

From your browser, open the above Network URL (`http://10.110.50.241:8501`).
