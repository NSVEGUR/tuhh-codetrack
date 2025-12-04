from setuptools import setup

setup(
    name='amllib',
    packages=['amllib',
              'amllib.activations',
              'amllib.initializers',
              'amllib.layers',
              'amllib.losses',
              'amllib.networks',
              'amllib.optimizers',
              'amllib.regularizers',
              'amllib.utils'],
    python_requires='>=3.9', #At least python 3.9 for type hints
    version='1.1',
    description='Machine learning library'\
                'for the lecture \"Advanced'\
                'Machine Learning\"',
    author='Jens-Peter M. Zemke, Jonas Grams',
    install_requires=['numpy',
                      'scipy',
                      'matplotlib']
)
