/*
 * This file is part of the uumpy project
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2019 Nicko van Someren
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// SPDX-License-Identifier: MIT

#include "py/mpconfig.h"

// Use the Micropython float type as the default uumpy array type
#if MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_NONE
#error uumpy requires Micropython to be compiled with floating point support
#elif MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_FLOAT
#define UUMPY_DEFAULT_TYPE 'f'
#elif MICROPY_FLOAT_IMPL == MICROPY_FLOAT_IMPL_DOUBLE
#define UUMPY_DEFAULT_TYPE 'd'
#else
#error Unknown floting point implementation type
#endif

// Feature support
#define UUMPY_ENABLE_HYPERBOLIC (1)
#define UUMPY_ENABLE_LINALG (1)
#define UUMPY_ENABLE_FFT (1)

// Time/space trade-off performance settings
// Include float-specfic implementations
#define UUMPY_SPEEDUP_FLOAT (1)
// Include regular integer-specfic implementations
#define UUMPY_SPEEDUP_INT (1)

