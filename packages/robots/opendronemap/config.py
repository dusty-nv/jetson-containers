
node = package.copy()

node['name'] = 'opendronemap:node'
node['dockerfile'] = 'Dockerfile.node'
node['depends'] = ['opendronemap:latest', 'nodejs']

package = [package, node]

