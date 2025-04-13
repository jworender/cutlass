from setuptools import setup, find_packages

setup(
    name='cutlass',
    version='0.5',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'xgboost',
        'pytest',
        'matplotlib',
        'scikit-learn'
    ],
    description='Encapsulates data, embeds data science methods, and implements the Lasso Logic Engine.',
    author='Jason Orender',
    author_email='jason@orender.net',
)
