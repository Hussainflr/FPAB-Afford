from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='enclib_cpu',
    ext_modules=[
        CppExtension('enclib_cpu', [
            'operator.cpp',
            'encoding_cpu.cpp',
            'syncbn_cpu.cpp'
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
