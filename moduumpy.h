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

#ifndef UUMPY_INCLUDED_MODUUMPY_H
#define UUMPY_INCLUDED_MODUUMPY_H

#include "py/obj.h"
#include "py/objarray.h"

#include "uumpy_config.h"

#define UUMPY_DTYPE_GUESS (0)
#define UUMPY_DTYPE_BOOL (1)
#define UUMPY_DTYPE_BYTE (2)
#define UUMPY_DTYPE_UBYTE (3)
#define UUMPY_DTYPE_INT (4)
#define UUMPY_DTYPE_UINT (5)
#define UUMPY_DTYPE_LONG (6)
#define UUMPY_DTYPE_ULONG (7)
#define UUMPY_DTYPE_FLOAT (8)
#define UUMPY_DTYPE_COMPLEX (10)

// We limit ourselves to 8 dimensions in n-D arrays. Having a limit
// allows us to have dim lists on the stack.
#define UUMPY_MAX_DIMS 8

// Each dimension of an n-D array has a length and stride
typedef struct _uumpy_dim_info {
    mp_int_t length;
    mp_int_t stride;
} uumpy_dim_info;

// The same structure is used for simple arrays and views
typedef struct _uumpy_obj_ndarray_t {
    mp_obj_base_t base;
    size_t dim_count : 8;
    size_t typecode : 8;
    size_t simple : 1;
    size_t free : (8 * sizeof(size_t) - 17); // Padding
    mp_int_t base_offset;
    void *data;
    uumpy_dim_info *dim_info;
} uumpy_obj_ndarray_t;

// This is the type definition
extern const mp_obj_type_t uumpy_type_ndarray;

extern uumpy_obj_ndarray_t *uumpy_array_from_value(const mp_obj_t value, char typecode);
extern uumpy_obj_ndarray_t *ndarray_new(char typecode, size_t dim_count, size_t *dims);
extern bool ndarray_compare_dimensions(uumpy_obj_ndarray_t *left_in, uumpy_obj_ndarray_t *right_in);
bool ndarray_broadcast(uumpy_obj_ndarray_t *left_in, uumpy_obj_ndarray_t *right_in,
                       uumpy_obj_ndarray_t **left_out, uumpy_obj_ndarray_t **right_out);

uumpy_obj_ndarray_t *ndarray_new_from_ndarray(mp_obj_t value_in, char typecode);
uumpy_obj_ndarray_t *ndarray_new_view(uumpy_obj_ndarray_t *source, size_t new_base,
                                      size_t new_dim_count, uumpy_dim_info *new_dims);
uumpy_obj_ndarray_t *ndarray_new_shaped_like(char typecode, uumpy_obj_ndarray_t *other, size_t trim_dims);

#endif // UUMPY_INCLUDED_MODUUMPY_H
