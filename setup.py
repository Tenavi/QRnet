import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qrnet",
    version="3.5.0",
    description="Neural network optimal feedback control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tenavi Nakamura-Zimmerer",
    author_email="tenakamu@ucsc.edu",
    packages=["qrnet"]
)
