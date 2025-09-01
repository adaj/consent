from setuptools import setup, find_packages

setup(
    name='consent',
    version='0.2.0',
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
        'tensorflow==2.19.1',
        'tensorflow-text',
        'tensorflow-hub',
        'wandb==0.21.3',
        'fire',
        'tqdm',
        'python-dotenv'
    ]
)
