# integer-wavelet-transform
Implementation of integer wavelet transform using lifting scheme.

## Usage

### Using CLI
After installing the package, a new CLI is exposed to your shell:

`> wavelets`

All flags are explained in the help menu, which can be shown by specifying the `-h` flag.

In general, it is important to specify if you want to decompose the image (`-d`) or reconstruct the original image from a decomposed image (`-r`).

### Running through code. 
Please take a look at the provided example in `example.py`. This shows how you can specify a wavelet and use to transform an image.

## Installation

### Using PyPI.org
The project is published on [PyPI.org](https://pypi.org/project/integer-wavelets/). To install the package with all required packages run:

`> pip install integer-wavelets`

### Installing manually
To build the project from source start by cloning the project:

`> git clone https://github.com/BrPetrus/integer-wavelet-transform`

Install poetry:

`> pip install poetry`

Now you can build the wheel files using:

`> poetry build`

This will build the project under the `dist` folder. Now you can install the package by running:

`> pip install dist/*`

## Generated files

## Specifying custom wavelets