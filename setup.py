from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='RepCONC',
    version='0.1.0',
    description='Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval',
    url='https://github.com/jingtaozhan/RepCONC',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=['repconc'],
    long_description=readme,
    install_requires=[
        'torch >= 1.9.0',
        'transformers >= 4.3.3',
        #'faiss-gpu == 1.7.1',#faiss should be installed manually
        'tensorboard >= 2.5.0',
        'boto3',
        'jpq@git+https://github.com/jingtaozhan/JPQ#egg=jpq'
    ],
)