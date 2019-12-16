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

#include <math.h>

#include "py/runtime.h"

#include "moduumpy.h"
#include "ufunc.h"
#include "uumath.h"

#define UUMATH_FUN_1(name) \
    STATIC mp_obj_t uumpy_math_ ## name(size_t n_args, const mp_obj_t *args, mp_map_t *kwargs) { \
        return uumpy_math_helper_1(MICROPY_FLOAT_C_FUN( name ), n_args, args, kwargs); \
    } \
    MP_DEFINE_CONST_FUN_OBJ_KW(uumpy_math_## name ## _obj, 1, uumpy_math_ ## name)

STATIC mp_obj_t uumpy_math_helper_1(uumpy_unary_float_func op_func,
                                    mp_uint_t n_args, const mp_obj_t *pos_args,
                                    mp_map_t *kw_args) {
    enum {
        ARG_x,
        ARG_out,
        ARG_dtype,
    };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_x,       MP_ARG_REQUIRED | MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_out,     MP_ARG_KW_ONLY  | MP_ARG_OBJ, {.u_obj = mp_const_none} },
        { MP_QSTR_dtype,   MP_ARG_KW_ONLY  | MP_ARG_OBJ, {.u_obj = mp_const_none} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    uumpy_obj_ndarray_t *src;    
    uumpy_obj_ndarray_t *dest;
    uumpy_universal_spec spec;
    char result_typecode = 0;
    
    // If the source is not an array then make one
    if (!mp_obj_is_type(args[ARG_x].u_obj, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        src = uumpy_array_from_value(args[ARG_x].u_obj, 'f');
    } else {
        src = MP_OBJ_TO_PTR(args[ARG_x].u_obj);
    }

    if (args[ARG_out].u_obj == mp_const_none) {
        if (args[ARG_dtype].u_obj != mp_const_none) {
            if (args[ARG_out].u_obj != mp_const_none) {
                mp_raise_ValueError("dtype and out arguments mutaully exclusive");
            }
            size_t type_len;
            const char *typecode_ptr = mp_obj_str_get_data(args[ARG_dtype].u_obj, &type_len);
            if (type_len != 1) {
                mp_raise_ValueError("Data type should be a single character code");
            }
            result_typecode = typecode_ptr[0];
        } else {
            result_typecode = 0;
        }
    } else {
        if (args[ARG_dtype].u_obj != mp_const_none) {
            mp_raise_ValueError("dtype and out arguments mutaully exclusive");
        } else {
            dest = MP_OBJ_TO_PTR(args[ARG_out].u_obj);
            result_typecode = dest->typecode;
        }
    }
    
    ufunc_find_unary_float_func_spec(src, &result_typecode, op_func, &spec);
    
    if (args[ARG_out].u_obj == mp_const_none) {
        // make a new array of the right size, with possible data type
        size_t dims[UUMPY_MAX_DIMS];
        for (size_t i=0; i < src->dim_count; i++) {
            dims[i] = src->dim_info[i].length;
        }
        // DEBUG_printf("Creating destination array\n");
        dest = ndarray_new(result_typecode, src->dim_count, dims);
    } else {
        dest = MP_OBJ_TO_PTR(&args[ARG_out]);
        // Broadcast input if necessary. It's slower than expanding the result but uses less memory.
        if (!ndarray_compare_dimensions(src, dest)) {
            if (ndarray_broadcast(dest, src, &dest, &src)) {
                mp_raise_ValueError("non-broadcastable output operand");
            }
        }
    }

    // Apply function
    if (ufunc_apply_unary(dest, src, &spec)) {
        return MP_OBJ_FROM_PTR(dest);
    } else {
        mp_raise_ValueError("math error");
    }
}

UUMATH_FUN_1(sin);
UUMATH_FUN_1(cos);
UUMATH_FUN_1(tan);
UUMATH_FUN_1(asin);
UUMATH_FUN_1(acos);
UUMATH_FUN_1(atan);

#if UUMPY_ENABLE_HYPERBOLIC
UUMATH_FUN_1(sinh);
UUMATH_FUN_1(cosh);
UUMATH_FUN_1(tanh);
UUMATH_FUN_1(asinh);
UUMATH_FUN_1(acosh);
UUMATH_FUN_1(atanh);
#endif

UUMATH_FUN_1(exp);
UUMATH_FUN_1(log);

