package['name'] = 'gptq-for-llama:cuda'
package['alias'] = 'gptq-for-llama'
package['build_args'] = {
    'GPTQ_FOR_LLAMA_REPO': 'oobabooga/GPTQ-for-LLaMa',
    'GPTQ_FOR_LLAMA_BRANCH': 'cuda',
}

gptq_for_llama_triton = package.copy()
gptq_for_llama_triton['name'] = 'gptq-for-llama:triton'
gptq_for_llama_triton['build_args'] = {
    'GPTQ_FOR_LLAMA_REPO': 'qwopqwop200/GPTQ-for-LLaMa',
    'GPTQ_FOR_LLAMA_BRANCH': 'triton',
}
gptq_for_llama_triton['depends'] = gptq_for_llama_triton['depends'] + ['openai-triton']

package = [package, gptq_for_llama_triton]