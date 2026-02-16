"""Package configuration for rna_fold."""
from setuptools import setup, find_packages

setup(
    name="rna_fold",
    version="0.1.0",
    description=(
        "Template-Based Modeling pipeline for RNA 3D structure prediction. "
        "Stanford RNA 3D Folding Challenge."
    ),
    author="Qasim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
    ],
)
