from jetson_containers import L4T_VERSION

package['build_args'] = {
    'L4T_VERSION_SHORT': f'{L4T_VERSION.major}.{L4T_VERSION.minor}'
}
