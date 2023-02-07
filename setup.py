# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 19:26
from os.path import abspath, join, dirname
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()
version = {}
with open(join(this_dir, "elit", "version.py")) as fp:
    exec(fp.read(), version)

setup(
    name='seq2seq-corenlp',
    version=version['__version__'],
    description='Unleashing the True Potential of Sequence-to-Sequence Models for Sequence Tagging and Structure Parsing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/emorynlp/seq2seq-corenlp',
    author='Han He',
    author_email='han.he@emory.edu',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        "Development Status :: 3 - Alpha",
        'Operating System :: OS Independent',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Topic :: Text Processing :: Linguistic"
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    install_requires=[
        'termcolor',
        'pynvml',
        'alnlp',
        'toposort==1.5',
        'transformers==4.9.2',
        'sentencepiece>=0.1.91'
        'torch>=1.6.0',
        'hanlp-common==0.0.11',
        'hanlp-trie==0.0.4',
        'hanlp-downloader',
        'tensorboardX==2.1',
        'penman==1.2.2',
        'networkx==2.8.8',
    ],
    python_requires='>=3.6',
)
