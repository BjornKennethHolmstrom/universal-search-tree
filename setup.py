from setuptools import setup, find_packages

setup(
    name="ust",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'networkx',
        'matplotlib'
    ],
)
