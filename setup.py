from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'flowformer',
    version = '1.0.0',
    author = 'Matthias Woedlinger',
    author_email = 'mwoedlinger@cvl.tuwien.ac.at',
    description = 'Applying transformers to cell data',
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages = find_packages(),
    install_requires = [
        'tqdm',
        'pandas',
        'tensorboard',
        'scikit-learn',
        'matplotlib',
        'torch',
        'torchvision',
        'flowmepy'
    ],
)
