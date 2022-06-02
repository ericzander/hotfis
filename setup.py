import setuptools

setuptools.setup(
    name = "hotfis",
    version = "0.1.2",
    description = "Create, alter, and visualize fuzzy inference systems in Python!",
    author = "Eric Zander",
    url = "https://github.com/ericzander/hotfis",
    packages = setuptools.find_packages(),
    license="LICENSE",
    install_requires = [
        "matplotlib>=3.5.1",
        "numpy>=1.22.3",
    ],
    python_requires = ">=3.8",
)
