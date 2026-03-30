from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="mlops_project",
    version="0.1",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)