

version_2 = package.copy()

version_2['name'] = 'jetson-utils:v2'

version_2['build_args'] = {
    'JETSON_UTILS_BRANCH': 'master',
    'JETSON_UTILS_CMAKE': "on",
    'JETSON_UTILS_PIP': "on",
}

python_only = package.copy()

python_only['name'] = 'jetson-utils:python'
python_only['depends'] = 'python'

python_only['build_args'] = {
    'JETSON_UTILS_BRANCH': 'master',
    'JETSON_UTILS_CMAKE': "off",
    'JETSON_UTILS_PIP': "on",
}

package['name'] = 'jetson-utils:v1'
package['alias'] = 'jetson-utils'

package['build_args'] = {
    'JETSON_UTILS_BRANCH': 'v1',
    'JETSON_UTILS_CMAKE': "on",
    'JETSON_UTILS_PIP': "off",
}

package = [package, version_2, python_only]

