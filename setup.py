from setuptools import setup, find_packages

setup(
    name='semantic_backprop',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'numpy',
        'openai',
        'tqdm',
        'wandb',
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
)