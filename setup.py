import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="luCART-lucamasserano", # Replace with your own username
    version="0.0.1",
    author="Luca Masserano",
    author_email="masserano.luca@gmail.com",
    description="A package implementing CART and Random Forests",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/36-750/assignments-lucamasserano",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        #"Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
