from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
extensions = []
for name in ('aq_lut', 'aq_lut_gqa', 'fused_linear_encode'):
    flags = ['-O3'] if name == 'aq_lut' else ['-O3', '--use_fast_math']
    extensions.append(CUDAExtension(
        'aq_kernels.' + name, [str(ROOT / 'aq_kernels' / (name + '.cu'))],
        extra_compile_args={'nvcc': flags}))
setup(name='laprune', version='0.1.0', packages=find_packages(),
      ext_modules=extensions, cmdclass={'build_ext': BuildExtension})
