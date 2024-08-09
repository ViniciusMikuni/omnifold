# setup.py
from setuptools import setup, find_packages

setup(
    name='unbinned_unfold',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'horovod==0.28.1',
        'matplotlib==3.8.2',
        'numpy==2.0.1',
        'PyYAML==6.0.1',
        'PyYAML==6.0.2',
        'setuptools==69.0.3',
        'tensorflow==2.15.0',
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
