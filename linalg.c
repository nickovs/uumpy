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

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "py/runtime.h"
#include "py/builtin.h"
#include "py/binary.h"
#include "py/objtuple.h"
#include "py/objexcept.h"

#include "moduumpy.h"
#include "linalg.h"
#include "ufunc.h"

#define ABS(x) MICROPY_FLOAT_C_FUN(fabs)(x)
#define FREXP(x, exp_p) MICROPY_FLOAT_C_FUN(frexp)(x, exp_p)

#if UUMPY_ENABLE_LINALG

const mp_obj_type_t uumpy_linalg_type_LinAlgError = {
    { &mp_type_type },
    .name = MP_QSTR_LinAlgError,
    .print = mp_obj_exception_print,
    .make_new = mp_obj_exception_make_new,
    .attr = mp_obj_exception_attr,
    .parent = &mp_type_Exception,
};


// Given a matrix of flaots, n wide and m high, m <=n, manipulate it
// into lower triangular form and return the product of the set of
// changes that the manipulations have on the determinant of the
// left-most m by m part of the matrix.  This MUST have a stride of n
// for the first dimension and 1 for the second doimension

// Swap two rows and optionally negate one of them
// The negate flag allows this to have no net effect on the determinant
STATIC void _swap_negate_rows(mp_float_t *data,
                              size_t width, size_t start,
                              size_t a_row_index, size_t b_row_index,
                              bool negate) {
    mp_float_t *row_a = data + (a_row_index * width);
    mp_float_t *row_b = data + (b_row_index * width);
    mp_float_t neg = negate ? -1.0 : 1.0;
    for (size_t i=start; i < width; i++) {
        mp_float_t tmp = row_a[i];
        row_a[i] = row_b[i];
        row_b[i] = tmp * neg;
    }
}

// Subtract a multiple of row a from row b so that the row b starts with a zero
STATIC void _subtract_to_zero(mp_float_t *data,
                              size_t width, size_t start,
                              size_t a_row_index, size_t b_row_index) {
    mp_float_t *row_a = data + (a_row_index * width);
    mp_float_t *row_b = data + (b_row_index * width);
    mp_float_t multiple = row_b[start] / row_a[start];
    
    if (multiple != 0) {
        row_b[start] = 0.0;
        for (size_t i = start+1; i < width; i++) {
            row_b[i] -= row_a[i] * multiple;
        }
    }    
}

STATIC void _divide_row(mp_float_t *data, size_t width,
                        size_t start, size_t row_index, mp_float_t d) {
    mp_float_t *row = data + (row_index * width);
    for (size_t i = start; i < width; i++) {
        row[i] /= d;
    }
}


// This function can be used to diagonalise or just to find row-echelon from
// Set norm to reduce leading diag to ones.
STATIC size_t _uumpy_linalg_reduce_array(uumpy_obj_ndarray_t *a, bool diag,
                                         bool norm, mp_float_t *det_change_out) {
    mp_float_t *data = (mp_float_t *) a->data;
    size_t width = a->dim_info[1].length;
    size_t height = a->dim_info[0].length;
    size_t x=0, y=0;
    mp_float_t det_change = 1.0;

    while (y < height && x < width) {
        int best_row = -1;
        int best_abs_exponent = 0x7fff;

        for (size_t j = y; j < height; j++) {
            mp_float_t v = data[x + j * width];

            if (ABS(v) < UUMPY_EPSILON) {
                continue;
            }

            if (v == 1.0) {
                best_row = j;
                break;
            }
            
            int row_abs_exp;
            FREXP(v, &row_abs_exp);
            row_abs_exp = abs(row_abs_exp);
            
            if (row_abs_exp < best_abs_exponent) {
                best_row = j;
                best_abs_exponent = row_abs_exp;
            }
        }

        if (best_row != -1) {
            if (best_row != y) {
                _swap_negate_rows(data, width, x, y, best_row, true);
            }
            if (diag) {
                for (size_t j = 0; j < height; j++) {
                    if (j != y) {
                        _subtract_to_zero(data, width, x, y, j);
                    }
                }
            } else {
                for (size_t j = y+1; j < height; j++) {
                    _subtract_to_zero(data, width, x, y, j);
                }
            }
            if (norm) {
                mp_float_t d = data[x + y * width];
                _divide_row(data, width, x, y, d);
                det_change /= d;
            }
            y += 1;
        }
        x += 1;
    }

    if (det_change_out != NULL) {
        *det_change_out = det_change;
    }
    
    return x;
}

STATIC mp_obj_t uumpy_linalg_re(mp_obj_t arg_in) {
    uumpy_obj_ndarray_t *o = uumpy_array_from_value(arg_in, UUMPY_DEFAULT_TYPE);
    
    // FIXME
    if (o->dim_count != 2) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "can only apply row echalon form to 2D matricies");
    }

    _uumpy_linalg_reduce_array(o, false, true, NULL);

    return MP_OBJ_FROM_PTR(o);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(uumpy_linalg_re_obj, uumpy_linalg_re);

STATIC mp_obj_t uumpy_linalg_det(mp_obj_t arg_in) {
    uumpy_obj_ndarray_t *o = uumpy_array_from_value(arg_in, UUMPY_DEFAULT_TYPE);
    
    // FIXME
    if (o->dim_count != 2) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "can only apply row echalon form to 2D matricies");
    }
    if (o->dim_info[0].length != o->dim_info[1].length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "det can only be applied to square matricies");
    }

    mp_float_t det_change;
    _uumpy_linalg_reduce_array(o, false, true, &det_change);
    
    return  mp_obj_new_float(1 / det_change);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(uumpy_linalg_det_obj, uumpy_linalg_det);

STATIC mp_obj_t uumpy_linalg_inv(mp_obj_t arg_in) {
    uumpy_obj_ndarray_t *o = uumpy_array_from_value(arg_in, UUMPY_DEFAULT_TYPE);

    // FIXME
    if (o->dim_count != 2) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "can only invert 2D matricies");
    }
    if (o->dim_info[0].length != o->dim_info[1].length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "inv can only be applied to square matricies");
    }

    size_t length = o->dim_info[0].length;
    size_t temp_dims[2];
    temp_dims[0] = length;
    temp_dims[1] = length * 2;
    uumpy_obj_ndarray_t *temp = ndarray_new(UUMPY_DEFAULT_TYPE, 2, temp_dims);

    uumpy_dim_info temp_view_dims[2];
    temp_view_dims[0] = temp->dim_info[0];
    temp_view_dims[1] = temp->dim_info[1];
    temp_view_dims[0].length = length;

    uumpy_obj_ndarray_t *temp_view = ndarray_new_view(temp, temp->base_offset, 2, temp_view_dims);
    uumpy_universal_spec copy_spec;
    
    ufunc_find_copy_spec(o, temp_view, NULL, &copy_spec);
    ufunc_apply_unary(temp_view, o, &copy_spec);

    mp_float_t *data = (mp_float_t *) temp->data;
    for (size_t j = 0; j < length; j++) {
        for (size_t i = 0; i < length; i++) {
            data[j * (length * 2) + i + length] = (i == j) ? 1.0 : 0.0;
        }
    }

    size_t end_column = _uumpy_linalg_reduce_array(temp, true, true, NULL);
    if (end_column != length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "singular matrix");
    }
    
    temp_view->base_offset = length;
    ufunc_find_copy_spec(temp_view, o, NULL, &copy_spec);
    ufunc_apply_unary(o, temp_view, &copy_spec);
    
    return MP_OBJ_FROM_PTR(o);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_1(uumpy_linalg_inv_obj, uumpy_linalg_inv);

STATIC mp_obj_t uumpy_linalg_solve(mp_obj_t a_in, mp_obj_t b_in) {
    uumpy_obj_ndarray_t *a = uumpy_array_from_value(a_in, UUMPY_DEFAULT_TYPE);
    uumpy_obj_ndarray_t *b = uumpy_array_from_value(b_in, UUMPY_DEFAULT_TYPE);

    // FIXME
    if (a->dim_count != 2 || b->dim_count != 1) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "can only solve single set of equations");
    }
    if (a->dim_info[0].length != a->dim_info[1].length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "equationa matrix must be square");
    }
    if (a->dim_info[0].length != b->dim_info[0].length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "dimensions dton't match");
    }

    size_t length = a->dim_info[0].length;
    size_t temp_dims[2];
    temp_dims[0] = length;
    temp_dims[1] = length + 1;
    uumpy_obj_ndarray_t *temp = ndarray_new(UUMPY_DEFAULT_TYPE, 2, temp_dims);

    uumpy_dim_info temp_view_dims[2];
    temp_view_dims[0] = temp->dim_info[0];
    temp_view_dims[1] = temp->dim_info[1];
    temp_view_dims[0].length = length;

    uumpy_obj_ndarray_t *temp_view = ndarray_new_view(temp, temp->base_offset, 2, temp_view_dims);
    uumpy_universal_spec copy_spec;
    
    ufunc_find_copy_spec(a, temp_view, NULL, &copy_spec);
    ufunc_apply_unary(temp_view, a, &copy_spec);

    temp_view->dim_count = 1;
    temp_view->base_offset = length;
    
    ufunc_find_copy_spec(b, temp_view, NULL, &copy_spec);
    ufunc_apply_unary(temp_view, b, &copy_spec);
    
    size_t end_column = _uumpy_linalg_reduce_array(temp, true, true, NULL);
    if (end_column != length) {
        mp_raise_msg(&uumpy_linalg_type_LinAlgError, "singular matrix");
    }
    
    temp_view->base_offset = length;
    ufunc_find_copy_spec(temp_view, b, NULL, &copy_spec);
    ufunc_apply_unary(b, temp_view, &copy_spec);
    
    return MP_OBJ_FROM_PTR(b);
}
STATIC MP_DEFINE_CONST_FUN_OBJ_2(uumpy_linalg_solve_obj, uumpy_linalg_solve);


STATIC const mp_rom_map_elem_t uumpy_linalg_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR_re), MP_ROM_PTR(&uumpy_linalg_re_obj) },
    { MP_ROM_QSTR(MP_QSTR_det), MP_ROM_PTR(&uumpy_linalg_det_obj) },
    { MP_ROM_QSTR(MP_QSTR_inv), MP_ROM_PTR(&uumpy_linalg_inv_obj) },
    { MP_ROM_QSTR(MP_QSTR_solve), MP_ROM_PTR(&uumpy_linalg_solve_obj) },
// inner
// outer
// QR
// SVD
// Cholesky
// eig
// eigvals
// norm
// trace
// lstsq
// pinv

    { MP_ROM_QSTR(MP_QSTR_LinAlgError), MP_ROM_PTR(&uumpy_linalg_type_LinAlgError) },
    
};
STATIC MP_DEFINE_CONST_DICT(uumpy_linalg_module_globals, uumpy_linalg_module_globals_table);

// Define module object.
const mp_obj_module_t uumpy_linalg_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&uumpy_linalg_module_globals,
};

#endif // UUMPY_ENABLE_LINALG
