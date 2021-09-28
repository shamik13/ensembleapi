from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read()
setup(
    name="ensembleapi",
    version="0.0.1",
    author="Shamik",
    author_email="shamik@nablas.com",
    url="https://github.com/shamik13/ensembleapi",
    py_modules=["runner", "app"],
    packages=find_packages(),
    install_requires=[requirements],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "ensembleapi=runner:cli",
        ],
    },
)
