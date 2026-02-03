
from jetson_containers import L4T_VERSION, CUDA_ARCHITECTURES

def build_cudf(version, arrow='arrow', repo='dusty-nv/cudf', requires=None, default=False):
    cudf = package.copy()

    cudf['name'] = f'cudf:{version}'
    cudf['group'] = 'rapids'
    cudf['notes'] = 'installed under `/usr/local`'

    cudf['build_args'] = {
        'CUDF_REPO': repo,
        'CUDF_VERSION': f'v{version}',
        'CUDF_CMAKE_CUDA_ARCHITECTURES': ';'.join([str(x) for x in CUDA_ARCHITECTURES]),
    }

    cudf['test'] = ['test_cudf.py', 'test_dask.py', 'test_csv.py']
    cudf['depends'] = ['cuda', 'cmake', 'python', 'cupy', 'numba', 'protobuf:apt']

    if L4T_VERSION.major >= 36:
        cudf['dockerfile'] = 'Dockerfile.jp6'
        cudf['depends'].extend(['cuda-python', arrow])
        cudf['test'].append('test_pandas.py')
    else:
        cudf['dockerfile'] = 'Dockerfile.jp5'

    if default:
        cudf['alias'] = 'cudf'

    if requires:
        cudf['requires'] = requires

    return cudf

package = [
    build_cudf('26.04.00', 'arrow:19.0.1', requires='>=36', default=True),
    build_cudf('26.04.00', requires='==35.*', default=True)
]
