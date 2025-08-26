
vla_mimicgen = package.copy()

vla_mimicgen['name'] = 'openvla-oft:mimicgen'
vla_mimicgen['depends'] = vla_mimicgen['depends'] + ['mimicgen']

package = [package, vla_mimicgen]
