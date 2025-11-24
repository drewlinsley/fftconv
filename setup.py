from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='flashfftconv',
        version='0.1.0',
        description='Fast FFT algorithms for convolutions',
        url='https://github.com/HazyResearch/flash-fft-conv',
        author='Dan Fu, Hermann Kumbong',
        author_email='danfu@cs.stanford.edu',
        license='Apache 2.0',
        packages=find_packages(),
        python_requires='>=3.8',
        install_requires=[
            'torch>=2.0.0',
            'numpy>=1.20.0',
            'einops>=0.6.0',
        ],
        extras_require={
            'test': [
                'pytest>=7.0.0',
            ],
        },
    )