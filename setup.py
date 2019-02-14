import setuptools

setuptools.setup(
    name="ermine",
    version="0.1",
    author="kitfactory",
    author_email="kitfactory@gmail.com",
    description="python package",
    long_description="ermine ...",
    long_description_content_type="text/markdown",
    url="https://github.com/kitfactory/ermine.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6.0",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['bin/ermine-web']
)
