
Container for [MLC LLM](https://github.com/mlc-ai/mlc-llm) project using Apache TVM Unity with CUDA, cuDNN, CUTLASS, FasterTransformer, and FlashAttention-2 kernels.

### Benchmarks

To quantize and benchmark a model, run the [`benchmark.sh`](benchmark.sh) script from the host (outside container)

```bash
HUGGINGFACE_TOKEN=hf_abc123def ./benchmark.sh meta-llama/Llama-2-7b-hf
```

This will run the quantization and benchmarking in the MLC container, and save the performance data to `jetson-containers/data/benchmarks/mlc.csv`.  If you are accessing a gated model, substitute your HuggingFace account's API key above.  Omitting the model will benchmark a default set of Llama models.  See [`benchmark.sh`](benchmark.sh) for various environment variables you can set.

```
AVERAGE OVER 3 RUNS, input=16, output=128
/data/models/mlc/0.1.0/Llama-2-7b-hf-q4f16_ft/params:  prefill_time 0.025 sec, prefill_rate 632.8 tokens/sec, decode_time 2.731 sec, decode_rate 46.9 tokens/sec
```

The prefill time is how long the model takes to process the input context before it can start generating output tokens.  The decode rate is the speed at which it generates output tokens.  These results are averaged over the number of prompts, minus the first warm-up.
