import setuptools

setuptools.setup(
    name="amortized_assimilation",
    version="0.0.1",
    author="Anonymized",
    author_email="Anonymized",
    description="Learned uncertainty-aware filters for assimilation noisy high dimensional observational data",
    url="Anonymized",
    packages=['amortized_assimilation'],
    install_requires=['torch>=1.3.1',
                        'matplotlib>=3.1.0',
                        'torchdiffeq>=0.0.1',
                        'numpy>=1.16.4'],
    classifiers=(
        "Programming Language :: Python :: 3"),)