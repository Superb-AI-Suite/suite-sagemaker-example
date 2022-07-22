import re, ast
from setuptools import setup, find_packages

setup(
    name='suite-detection',
    version='0.2.0',
    author='Kye-Hyeon Kim',
    author_email='khkim@superb-ai.com',
    packages=find_packages(),
    install_requires=[
        'spb-cli>=0.13.0',
        'pycocotools>=2.0.4',
    ],
)
