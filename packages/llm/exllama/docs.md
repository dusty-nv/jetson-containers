
This is using the https://github.com/jllllll/exllama fork of https://github.com/turboderp/exllama  

It's found under `/opt/exllama`, and the pip wheel is at `/opt/exllama-*.whl` and has been installed (with the CUDA kernels already built)

### Inference Benchmark

Substitute the GPTQ model from [HuggingFace Hub](https://huggingface.co/models?search=gptq) that you want to run (see [exllama compatible models](https://github.com/turboderp/exllama/blob/master/doc/model_compatibility.md))

```bash
./run.sh --workdir=/opt/exllama $(./autotag exllama) /bin/bash -c \
  'python3 test_benchmark_inference.py --perf --validate -d $(huggingface-downloader TheBloke/Llama-2-7B-GPTQ)'
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

### Memory Usage

| Model                                                                           | Memory (MB) |
|---------------------------------------------------------------------------------|:-----------:|
| [`TheBloke/Llama-2-7B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-7B-GPTQ)   |    5,200    |
| [`TheBloke/Llama-2-13B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-13B-GPTQ) |    9,135    |
| [`TheBloke/LLaMA-30b-GPTQ`](https://huggingface.co/TheBloke/LLaMA-30b-GPTQ)     |   20,206    |
| [`TheBloke/Llama-2-70B-GPTQ`](https://huggingface.co/TheBloke/Llama-2-70B-GPTQ) |   35,462    |

