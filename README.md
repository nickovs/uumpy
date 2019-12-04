# uumpy: a subset of numpy for Micropython

**uumpy** is a *'micro'* implementation of parts of [numpy](https://numpy.org) for
[Micropython](https://micropython.org). It aims to provide an efficient and compact
implementation of arithmetic on matricies and multi-dimensional data types, as well
as basic linear algebra. In due course the aim is also to support FFT functions.

## Usage

As far as possible *uumpy* should work just like *numpy*. As a result you should just be able
to use `import uumpy as np` and use the `np` module as you would with `numpy`.


## Compilation

This code forms an 'external module' for Micropython. Documentation about how to
make use of external modules can be found in the [Micropython documentation](https://docs.micropython.org/en/latest/develop/cmodules.html).



