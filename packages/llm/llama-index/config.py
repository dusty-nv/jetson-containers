# make a samples variant of the container
samples = package.copy()

samples['name'] = 'llama-index:samples'
samples['dockerfile'] = 'Dockerfile.samples'
samples['depends'] = ['llama-index:main', 'jupyterlab']

del samples['alias']

package = [package, samples]