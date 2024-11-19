# From this directory:
#
# $ pip install -e .
#
# -e installs in editable mode so that changes in the
# code will be reflected immediately without reinstalling.

from setuptools import setup, find_packages

setup(
    name="densjax",
    version="0.1",
    packages=find_packages(),
)
