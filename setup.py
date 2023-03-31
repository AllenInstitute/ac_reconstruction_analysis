from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='acanalysis',


    description='Axonal connectomics reconstruction analysis',
    long_description='Tools for processing axonal connectomics reconstructions',

    # Author details
    author='Russel Torres',
    author_email='',

    #requirements
    install_requires=required,

    packages=find_packages()
)
