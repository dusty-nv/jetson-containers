
AutoGPTQ from https://github.com/PanQiWei/AutoGPTQ (installed under `/opt/AutoGPTQ`)

### Inference Benchmark

Substitute the GPTQ model from [HuggingFace Hub](https://huggingface.co/models?search=gptq) that you want to run:

```bash
./run.sh --workdir=/opt/AutoGPTQ/examples/benchmark/ $(./autotag auto_gptq) \
	python3 generation_speed.py --model_name_or_path TheBloke/LLaMA-7b-GPTQ --use_safetensors --max_new_tokens=128
```

If you get the error `Exllama kernel does not support query/key/value fusion with act-order`, try adding the `--no_inject_fused_attention` flag.
