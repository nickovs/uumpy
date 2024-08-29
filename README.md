# uumpy: a subset of numpy for Micropython

**uumpy** is a *'micro'* implementation of parts of [numpy](https://numpy.org) for
[Micropython](https://micropython.org). It aims to provide an efficient and compact
implementation of arithmetic on matrices and multidimensional data types, as well
as basic linear algebra. In due course the aim is also to support FFT functions.

## Motivation

Why on Earth would anyone want to put a huge package like *numpy* onto a microcontroller?
In general, you would not. That's why this is not a port of *numpy* but instead a new
implementation of a subset of the functionality. That subset is geared towards making it easy
and efficient to implement the sorts of algorithms that people *do* want to put on a
microcontroller. For instance, a microcontroller that has to make decisions based on noisy
sensor data from a variety of sensors might need to implement a [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)
to get the best guess as to its state. Implementing this sort of filter is vastly easier when
matrix operations can be expressed cleanly. Similarly performing
[Fourier transforms](https://en.wikipedia.org/wiki/Fourier_transform) and
[convolutions](https://en.wikipedia.org/wiki/Fourier_transform) can be important for extracting
the relevant parts of signals coming from sensors in control systems.

One of the goals of *uumpy* is to allow developers to tune the size of the implementation to
their needs. Large but self-contained blocks of functionality should be optional, so that they
don't need to be included if they are not needed. Where there are speed/size trade-offs to be
made it should be possible for these to be adjusted at compile-time. If math on multidimensional
arrays is all that is needed then *uumpy* can be pretty compact; when data-type specific math
code makes use of fully unrolled loops it can be fast. You choose.

## Usage

As far as possible *uumpy* should work just like *numpy*. As a result you should just be able
to use `import uumpy as np` and use the `np` module as you would with `numpy`.


## Compilation

This code forms an 'external module' for Micropython. Documentation about how to
make use of external modules can be found in the [Micropython documentation](https://docs.micropython.org/en/latest/develop/cmodules.html).


## Release status

This code is currently still in a pretty early state. It supports core
matrix and multidimensional array math operations a some basic linear
algebra but is currently missing most of the more esoteric functions
(although the foundations for them are mostly laid). As it stands it
is useful for making code that implements matrix maths more readable
and for solving simple linear systems but is otherwise incomplete.



