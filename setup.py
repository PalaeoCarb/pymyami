from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    package_data={'myami': ['parameters/*.json', 'parameters/*.csv', 'parameters/*.py']},
    include_package_data=True,
    zip_safe=True)
