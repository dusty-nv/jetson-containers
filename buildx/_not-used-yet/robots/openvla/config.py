
vla_mimicgen = package.copy()

vla_mimicgen['name'] = 'openvla:mimicgen'
vla_mimicgen['depends'] = vla_mimicgen['depends'] + ['mimicgen']

package = [package, vla_mimicgen]
