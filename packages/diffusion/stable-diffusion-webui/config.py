from jetson_containers import L4T_VERSION

if L4T_VERSION.major < 36:
    # JP5 - subsequent versions of A1111 incompatible with Python 3.8
    package['build_args'] = {
        'STABLE_DIFFUSION_WEBUI_REF': 'refs/tags/v1.6.0',
        'STABLE_DIFFUSION_WEBUI_SHA': 'v1.6.0'
    }
else:
    package['build_args'] = {
        'STABLE_DIFFUSION_WEBUI_SHA': 'master',
        'STABLE_DIFFUSION_WEBUI_REF': 'refs/heads/master',
    }
