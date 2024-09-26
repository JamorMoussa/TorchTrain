from setuptools import setup, find_packages

setup(
    name='torch_train',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch"
    ],
    # Metadata
    author='Moussa JAMOR',
    author_email='moussajamorsup@gmail.com',
    description='torch_utils is a PyTorch extension designed for training and building deep learning models.',
    url='https://github.com/JamorMoussa/torch_train',
)