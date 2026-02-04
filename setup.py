from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='chack-tools',
    version='0.1.0',
    description='Reusable tool library for Chack-compatible agents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Carlos Polop',
    packages=find_packages(include=['chack_tools', 'chack_tools.*']),
    python_requires='>=3.9',
    install_requires=[
        'requests>=2.31.0',
        'langchain-core>=0.3.0',
        'pypdf>=4.0.0',
    ],
    extras_require={
        'openai_agents': ['openai-agents>=0.2.0'],
    },
)
