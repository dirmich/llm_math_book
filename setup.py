# setup.py: local editable install metadata for llm_math
from setuptools import setup, find_packages

setup(
    name="llm_math",
    version="0.1.0",
    description="LLM Math Book utilities",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
        "scipy>=1.10",
    ],
    python_requires=">=3.9",
)
