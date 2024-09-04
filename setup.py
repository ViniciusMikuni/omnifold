# setup.py
from setuptools import setup, find_packages

setup(
    name='omnifold',
    version='0.1.11',
    packages=find_packages(),
    install_requires=[
        'horovod==0.28.1',
        'matplotlib',
        'numpy',
        'PyYAML',
        'setuptools',
        'tensorflow',
    ],
    author='Vinicius Mikuni',
    author_email='vmikuni@lbl.gov',
    description='OmniFold, a library to perform unbinned and high-dimensional unfolding for HEP.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ViniciusMikuni/omnifold',  # Update with your GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
