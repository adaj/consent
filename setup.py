from setuptools import setup, find_packages

setup(
    name='consent',
    version='0.1.4',
    description='Automated coding of chats using contextual information and sentence encoding',
    author='Adelson de Araujo',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['consent'],
    install_requires=[
        'pandas',
        'scikit-learn',
        'matplotlib',
        'openpyxl',
        'tensorflow==2.5.0',
        'tensorflow-text',
        'tensorflow-hub',
        'tensorflow_addons',
        'wandb==0.12.11',
        'fire',
        'tqdm'
    ]
)
