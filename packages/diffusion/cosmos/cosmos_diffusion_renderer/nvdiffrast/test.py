#!/usr/bin/env python3
print('testing nvdiffrast...')

try:
    import nvdiffrast.torch as dr
except:
    print('nvdiffrast not found!')
    dr = None

print('nvdiffrast OK\n')
