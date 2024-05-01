# make a samples variant of the container
samples = package.copy()

samples['name'] = 'langchain:samples'
samples['dockerfile'] = 'Dockerfile.samples'
samples['depends'] = ['langchain:main', 'jupyterlab']

del samples['alias']

package = [package, samples]