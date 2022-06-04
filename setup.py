from setuptools import setup, find_packages

setup(
    name='repconc',
    version='0.2.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    description='Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval',
    url='https://github.com/jingtaozhan/RepCONC',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    author='Jingtao Zhan',
    author_email='jingtaozhan@gmail.com',
    install_requires=[
        'torch >= 1.10.1',
        'transformers >= 4.19.2',
        #'faiss-gpu == 1.7.1',#faiss should be installed manually
        'GradCache@git+https://github.com/luyug/GradCache#egg=GradCache'
    ],
)