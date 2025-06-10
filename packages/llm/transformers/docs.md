
The HuggingFace [Transformers](https://huggingface.co/docs/transformers/index) library supports a wide variety of NLP and vision models with a convenient API, and is used by many of the other LLM packages.  There are a large number of models that it's compatible with on [HuggingFace Hub](https://huggingface.co/models).

> [!NOTE]  
> If you wish to use Transformer's integrated [bitsandbytes](https://huggingface.co/docs/transformers/main_classes/quantization#bitsandbytes-integration) quantization (`load_in_8bit/load_in_4bit`) or [AutoGPTQ](https://huggingface.co/docs/transformers/main_classes/quantization#autogptq-integration) quantization, run these containers instead which include those respective libraries installed on top of Transformers:
>   * [`auto_gptq`](/packages/llm/auto_gptq) (depends on Transformers)
>   * [`bitsandbytes`](/packages/llm/bitsandbytes) (depends on Transformers)

### Text Generation Benchmark

Substitute the [text-generation model](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) that you want to run (it should be a CausalLM model like GPT, Llama, ect)

```bash
./run.sh $(./autotag transformers) \
   huggingface-benchmark.py --model=gpt2
```
> If the model repository is private or requires authentication, add `--env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN>`

By default, the performance is measured for generating 128 new output tokens (this can be set with `--tokens=N`)

The prompt can be changed with `--prompt='your prompt here'`

#### Precision / Quantization

Use the `--precision` argument to enable quantization (options are: `fp32` `fp16` `fp4` `int8`, default is: `fp16`)

If you're using `fp4` or `int8`, run the [`bitsandbytes`](/packages/llm/bitsandbytes) container as noted above, so that bitsandbytes package is installed to do the quantization.  It's expected that 4-bit/8-bit quantization is slower through Transformers than FP16 (while consuming less memory) - see [here](https://huggingface.co/docs/transformers/main_classes/quantization) for more info.

Other libraries like [`exllama`](/packages/llm/exllama), [`awq`](/packages/llm/awq), and [`AutoGPTQ`](/packages/llm/auto-gptq) have custom CUDA kernels and more efficient quantized performance.

#### Llama2

* First request access from https://ai.meta.com/llama/
* Then create a HuggingFace account, and request access to one of the Llama2 models there like https://huggingface.co/meta-llama/Llama-2-7b-hf (doing this will get you access to all the Llama2 models)
* Get a User Access Token from https://huggingface.co/settings/tokens

```bash
./run.sh --env HUGGINGFACE_TOKEN=<YOUR-ACCESS-TOKEN> $(./autotag transformers) \
   huggingface-benchmark.py --model=meta-llama/Llama-2-7b-hf
```
