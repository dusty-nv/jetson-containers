# make a samples variant of the container
myst = package.copy()

myst['name'] = 'jupyterlab:myst'
myst['dockerfile'] = 'Dockerfile.myst'
myst['depends'] = ['jupyterlab:main']

del myst['alias']

package = [package, myst]