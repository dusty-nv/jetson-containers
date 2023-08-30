import copy

samples = copy.deepcopy(package)

samples['name'] = 'langchain:samples'
samples['depends'].append('jupyterlab')

package = [package, samples]