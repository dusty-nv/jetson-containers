
The HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) library supports a wide variety of NLP and vision models with a convenient API, that many of the other LLM packages have adopted.

### Text Generation Benchmark

Substitute the [text-generation model](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) from [HuggingFace Hub](https://huggingface.co/models?search=gptq) that you want to run (it should be a CausalLM model like GPT, Llama, ect)

```bash
./run.sh $(./autotag exllama) huggingface-benchmark.py --model=gpt2
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

By default, the performance is measured for generating 128 new output tokens (this can be set with `--tokens=N`)

#### Precision / Quantization

You can change the precision used and enable quantization with the `--precision` argument (options are: `fp32` `fp16` `fp4` `int8`)

The default is `fp16` - on JetPack 5, the [`bitsandbytes`](/packages/llm/bitsandbytes) package is included in the container to enable 4-bit/8-bit quantization through the Transformers API.  It's expected that 4-bit/8-bit quantization is slower through Transformers than FP16 (while consuming less memory).  Other libraries like [`exllama`](/packages/llm/exllama), [`awq`](/packages/llm/awq), and [`AutoGPTQ`](/packages/llm/auto-gptq) have custom CUDA kernels and more efficient quantized performance. 

#### Llama2

* First request access from https://ai.meta.com/llama/
* Then create a HuggingFace account, and request access to one of the Llama2 models there like https://huggingface.co/meta-llama/Llama-2-7b-hf (doing this will get you access to all the Llama2 models)
* Get a User Access Token from https://huggingface.co/settings/tokens
* In your terminal, run `export HUGGINGFACE_TOKEN=<COPY-TOKEN-HERE>`

```bash
./run.sh --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN $(./autotag exllama) \
  huggingface-benchmark.py --model=meta-llama/Llama-2-7b-hf
```
