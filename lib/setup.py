from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name='convsim',
        ext_modules=[
            CUDAExtension('convsim', [
                    'conv_sim.cpp',
                    'conv_sim_kernel.cu',
                ])
            ],
        cmdclass={
            'build_ext' : BuildExtension
            }
)
