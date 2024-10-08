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

// Include required definitions first.

#include <math.h>

#include "py/obj.h"
#include "py/runtime.h"
#include "py/builtin.h"
#include "py/binary.h"
#include "py/objtuple.h"

#include "moduumpy.h"
#include "ufunc.h"
#include "uumath.h"
#include "linalg.h"
#include "reductions.h"

static mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t lhs_in, mp_obj_t rhs_in);
static uumpy_obj_ndarray_t *ndarray_new_0d(mp_obj_t value, char typecode);

extern const mp_obj_type_t uumpy_type_ndarray;

bool uumpy_util_get_list_tuple(mp_obj_t value, mp_int_t *len, mp_obj_t **items) {
    if (mp_obj_is_type(value, MP_OBJ_FROM_PTR(&mp_type_list))) {
        mp_obj_list_t *list_ptr = MP_OBJ_TO_PTR(value);
        *len = list_ptr->len;
        *items = list_ptr->items;
        return true;
    } else if (mp_obj_is_type(value, MP_OBJ_FROM_PTR(&mp_type_tuple))) {
        mp_obj_tuple_t *tuple_ptr = MP_OBJ_TO_PTR(value);
        *len = tuple_ptr->len;
        *items = tuple_ptr->items;
        return true;
    } else {
        return false;
    }
}

static void ndarray_dot_helper_1d(uumpy_multiply_accumulate mac_fn, mp_int_t lhs_depth, mp_int_t rhs_depth,
                                  uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                  uumpy_obj_ndarray_t *lhs, mp_int_t lhs_offset,
                                  uumpy_obj_ndarray_t *rhs, mp_int_t rhs_offset) {
    // DEBUG_printf("MAC 1d, lhs_depth=%d, rhs_depth=%d, dest_offset = %d, lhs_offset=%d, rhs_offset=%d\n",
    //              lhs_depth, rhs_depth, dest_offset, lhs_offset, rhs_offset);

    if (lhs_depth == lhs->dim_count-2) {
        for (mp_int_t i=0; i < lhs->dim_info[lhs_depth].length; i++) {
            mac_fn(dest, dest_offset,
                   lhs, lhs_offset, lhs_depth+1,
                   rhs, rhs_offset, rhs_depth);
            dest_offset += dest->dim_info[lhs_depth].stride;
            lhs_offset += lhs->dim_info[lhs_depth].stride;
            // Don't advance rhs!
        }
    } else {
        for (mp_int_t i=0; i < dest->dim_info[lhs_depth].length; i++) {
            ndarray_dot_helper_1d(mac_fn, lhs_depth+1, rhs_depth,
                                  dest, dest_offset,
                                  lhs, lhs_offset,
                                  rhs, rhs_offset);
            dest_offset += dest->dim_info[lhs_depth].stride;
            lhs_offset += lhs->dim_info[lhs_depth].stride;
            // Don't advance rhs!
        }
    }
}

static void ndarray_dot_helper_Nd(uumpy_multiply_accumulate mac_fn, mp_int_t depth,
                                  uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                  uumpy_obj_ndarray_t *lhs, mp_int_t lhs_offset,
                                  uumpy_obj_ndarray_t *rhs, mp_int_t rhs_offset) {
    // In this function 'depth' refers to the depth of the rhs input
    mp_int_t dest_depth = depth + lhs->dim_count - 1;
    // DEBUG_printf("MAC Nd, depth=%d, dest_depth=%d, dest_offset = %d, lhs_offset=%d, rhs_offset=%d\n",
    //              depth, dest_depth, dest_offset, lhs_offset, rhs_offset);

    if (depth == rhs->dim_count - 2) {
        for (mp_int_t i=0; i < rhs->dim_info[depth+1].length; i++) {
            ndarray_dot_helper_1d(mac_fn, 0, depth,
                                  dest, dest_offset,
                                  lhs, lhs_offset,
                                  rhs, rhs_offset);
            dest_offset += dest->dim_info[dest_depth].stride;
            // Don't advance lhs_depth
            rhs_offset += rhs->dim_info[depth+1].stride;
        }
    } else {
        for (mp_int_t i=0; i < rhs->dim_info[depth].length; i++) {
            ndarray_dot_helper_Nd(mac_fn, depth+1,
                                  dest, dest_offset,
                                  lhs, lhs_offset,
                                  rhs, rhs_offset);
            dest_offset += dest->dim_info[dest_depth].stride;
            // Don't advance lhs_depth
            rhs_offset += rhs->dim_info[depth].stride;
        }
    }
}

static uumpy_obj_ndarray_t *ndarray_dot_impl(uumpy_obj_ndarray_t *lhs, uumpy_obj_ndarray_t *rhs) {
    // If either a or b is 0-D (scalar), it is equivalent to multiply.
    if (lhs->dim_count == 0 || rhs->dim_count == 0) {
        return MP_OBJ_TO_PTR(ndarray_binary_op(MP_BINARY_OP_MULTIPLY,
                                               MP_OBJ_FROM_PTR(lhs),
                                               MP_OBJ_FROM_PTR(rhs)));
    }

    uumpy_obj_ndarray_t *result = NULL;

    // TODO: Select function based on type
    uumpy_multiply_accumulate mac_fn = ufunc_mul_acc_fallback;
    // TODO: This will need to change when we add complex numbers
    char result_typecode = (lhs->typecode == UUMPY_DEFAULT_TYPE || rhs->typecode == UUMPY_DEFAULT_TYPE) ? UUMPY_DEFAULT_TYPE : 'i';
    mp_int_t dims[UUMPY_MAX_DIMS];

    if (lhs->dim_count == 1 && rhs->dim_count == 1) {
        // If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
        result = ndarray_new_0d(MP_OBJ_NEW_SMALL_INT(0), result_typecode);
        mac_fn(result, 0, lhs, lhs->base_offset, 0, rhs, rhs->base_offset, 0);
    } else if (rhs->dim_count == 1) {
        // If A is an N-D array and B is a 1-D array, it is a sum product over the last axis of A and B.
        if (lhs->dim_info[lhs->dim_count-1].length != rhs->dim_info[0].length) {
            mp_raise_ValueError(MP_ERROR_TEXT("incompatible dimensions"));
        }

        for (mp_int_t i=0; i < lhs->dim_count-1; i++) {
            dims[i] = lhs->dim_info[i].length;
        }

        result = ndarray_new_shaped_like(result_typecode, lhs, 1);
        ndarray_dot_helper_1d(mac_fn, 0, 0,
                             result, result->base_offset,
                             lhs, lhs->base_offset,
                             rhs, rhs->base_offset);
    } else {
        // If both a and b are 2-D arrays, it is matrix multiplication. This is a special case of:
        // If A is an N-D array and B is an M-D array (where M>=2), it is a sum product over the
        // last axis of A and the second-to-last axis of B.
        // dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

        if (lhs->dim_count + rhs->dim_count - 2 > UUMPY_MAX_DIMS) {
            mp_raise_ValueError(MP_ERROR_TEXT("result has too many dimensions"));
        }
        if (lhs->dim_info[lhs->dim_count-1].length != rhs->dim_info[rhs->dim_count-2].length) {
            mp_raise_ValueError(MP_ERROR_TEXT("incompatible dimensions"));
        }

        for (mp_int_t i=0; i < lhs->dim_count - 1; i++) {
            dims[i] = lhs->dim_info[i].length;
        }
        for (mp_int_t i=0; i < rhs->dim_count - 2; i++) {
            dims[i + lhs->dim_count - 1] = rhs->dim_info[i].length;
        }
        dims[lhs->dim_count + rhs->dim_count - 3] = rhs->dim_info[rhs->dim_count - 1].length;

        result = ndarray_new(result_typecode, lhs->dim_count + rhs->dim_count - 2, dims);

        ndarray_dot_helper_Nd(mac_fn, 0,
                              result, result->base_offset,
                              lhs, lhs->base_offset,
                              rhs, rhs->base_offset);
    }

    return result;
}

static mp_obj_t ndarray_dot(mp_obj_t lhs_in, mp_obj_t rhs_in) {
    uumpy_obj_ndarray_t *lhs;
    uumpy_obj_ndarray_t *rhs;
    uumpy_obj_ndarray_t *result;
    char typecode = UUMPY_DEFAULT_TYPE;

    if (mp_obj_is_type(rhs_in, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        rhs = MP_OBJ_TO_PTR(rhs_in);
        typecode = rhs->typecode;
    }

    if (!mp_obj_is_type(lhs_in, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        lhs = uumpy_array_from_value(lhs_in, typecode);
    } else {
        lhs = MP_OBJ_TO_PTR(lhs_in);
    }
    if (!mp_obj_is_type(rhs_in, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        rhs = uumpy_array_from_value(rhs_in, lhs->typecode);
    } else {
        rhs = MP_OBJ_TO_PTR(rhs_in);
    }

    result = ndarray_dot_impl(lhs, rhs);

    return MP_OBJ_FROM_PTR(result);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ndarray_dot_obj, ndarray_dot);


static void ndarray_print_helper(const mp_print_t *print, mp_int_t typecode,
                                 void *data, mp_int_t base_index,
                                 mp_int_t n_dims, uumpy_dim_info *dim_info) {
    mp_int_t length = dim_info[0].length;
    mp_int_t stride = dim_info[0].stride;

    // DEBUG_printf(" Length=%d, stride=%d\n", length, stride);

    mp_print_str(print, "[");
    for(mp_int_t i=0; i < length; i++, base_index += stride) {
        if (n_dims == 1) {
            mp_obj_print_helper(print,
                                mp_binary_get_val_array(typecode, data, base_index),
                                PRINT_REPR);
        } else {
            ndarray_print_helper(print, typecode,
                                 data, base_index,
                                 n_dims-1, dim_info+1);
        }
        if (i != length-1) {
            mp_print_str(print, ", ");
        }
    }
    mp_print_str(print, "]");
}

static void ndarray_print(const mp_print_t *print, mp_obj_t o_in, mp_print_kind_t kind) {
    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(o_in);
    mp_print_str(print, "ndarray(");

    if (o->dim_count == 0) {
        mp_obj_print_helper(print,
                            mp_binary_get_val_array(o->typecode, o->data, o->base_offset),
                            PRINT_REPR);

    } else {
        ndarray_print_helper(print, o->typecode,
                             o->data, o->base_offset,
                             o->dim_count, o->dim_info);
    }
    mp_printf(print, ", dtype='%c')", o->typecode);
}

uumpy_obj_ndarray_t *ndarray_new(char typecode, mp_int_t dim_count, mp_int_t *dims) {
    int typecode_size = mp_binary_get_size('@', typecode, NULL);

    uumpy_obj_ndarray_t *o = m_new_obj(uumpy_obj_ndarray_t);
    o->base.type = &uumpy_type_ndarray;
    o->typecode = typecode;
    o->dim_count = dim_count;
    // Brand new, so it counts as 'simple'
    o->simple = 1;
    o->free = 0;
    o->base_offset = 0;
    o->dim_info = m_new(uumpy_dim_info, dim_count);

    mp_int_t total_count = 1;

    for (mp_int_t i = dim_count-1; i >= 0; i--) {
        o->dim_info[i].length = dims[i];
        o->dim_info[i].stride = total_count;
        total_count *= dims[i];
    }

    // DEBUG_printf("Allocating new array type %c, dim_count=%d, total_count=%d\n", typecode, dim_count, total_count);
    
    o->data = m_new(byte, typecode_size * total_count);

    return o;
}

uumpy_obj_ndarray_t *ndarray_new_shaped_like(char typecode, uumpy_obj_ndarray_t *other,
                                             mp_int_t trim_dims) {
    mp_int_t dim_count = other->dim_count - trim_dims;
    mp_int_t dims[UUMPY_MAX_DIMS];

    for(mp_int_t i = 0; i < dim_count; i++) {
        dims[i] = other->dim_info[i].length;
    }

    return ndarray_new(typecode, dim_count, dims);
}

uumpy_obj_ndarray_t *ndarray_new_view(uumpy_obj_ndarray_t *source, mp_int_t new_base,
                                      mp_int_t new_dim_count, uumpy_dim_info *new_dims) {
    uumpy_obj_ndarray_t *o = m_new_obj(uumpy_obj_ndarray_t);
    o->base.type = &uumpy_type_ndarray;
    o->dim_count = new_dim_count;
    o->typecode = source->typecode;
    o->simple = 0; // We could improve on this...
    o->free = 0;
    o->base_offset = new_base;
    o->dim_info = m_new(uumpy_dim_info, new_dim_count);
    for (mp_int_t i=0; i < new_dim_count; i++) {
        o->dim_info[i] = new_dims[i];
    }
    o->data = source->data;
    return o;
}

uumpy_obj_ndarray_t *ndarray_new_from_ndarray(mp_obj_t value_in, char typecode) {
    uumpy_obj_ndarray_t *value = MP_OBJ_TO_PTR(value_in);
    uumpy_universal_spec copy_spec;

    ufunc_find_copy_spec(value, NULL, &typecode, &copy_spec);

    uumpy_obj_ndarray_t *o = ndarray_new_shaped_like(typecode, value, 0);

    ufunc_apply_unary(o, value, &copy_spec);

    return o;
}

static uumpy_obj_ndarray_t *ndarray_new_1d_from_iterable(mp_obj_t value, mp_int_t len, char typecode) {
    uumpy_obj_ndarray_t *o = ndarray_new(typecode, 1, &len);
    mp_obj_t iterable = mp_getiter(value, NULL);
    mp_obj_t item;
    mp_int_t i = 0;
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        if (i >= len) {
            mp_raise_ValueError(MP_ERROR_TEXT("Too many items from iterable"));
        }
        mp_binary_set_val_array(typecode, o->data, i++, item);
    }
    return o;
}

static uumpy_obj_ndarray_t *ndarray_new_0d(mp_obj_t value, char typecode) {
    uumpy_obj_ndarray_t *o = m_new_obj(uumpy_obj_ndarray_t);
    int typecode_size = mp_binary_get_size('@', typecode, NULL);

    o->base.type = &uumpy_type_ndarray;
    o->dim_count = 0;
    o->typecode = typecode;
    o->simple = 0;
    o->free = 0;
    o->base_offset = 0;

    o->dim_info = NULL;
    o->data = m_new(byte, typecode_size);

    mp_binary_set_val_array(o->typecode, o->data, 0, value);

    return o;
}

static mp_obj_t ndarray_get_obj_or_0d(uumpy_obj_ndarray_t *o) {
    if (o->dim_count == 0) {
        return mp_binary_get_val_array(o->typecode, o->data, o->base_offset);
    } else {
        return MP_OBJ_FROM_PTR(o);
    }
}

static void ndarray_copy_list_tuple(mp_obj_t value, uumpy_obj_ndarray_t *target, mp_int_t depth, mp_int_t offset) {
    bool last_dim = (depth == (target->dim_count));

    if (!value || value == mp_const_none) {
        mp_raise_ValueError(MP_ERROR_TEXT("can't assign None to array"));
    }


    mp_int_t length;
    mp_obj_t *items;

    if (uumpy_util_get_list_tuple(value, &length, &items)) {
        if (last_dim ||
            length != target->dim_info[depth].length) {
            mp_raise_ValueError(MP_ERROR_TEXT("incompatible shape"));
        }
        for (mp_int_t i=0; i < length; i++) {
            ndarray_copy_list_tuple(items[i], target, depth+1, offset);
            offset += target->dim_info[depth].stride;
        }
    } else {
        if (!last_dim) {
            mp_raise_ValueError(MP_ERROR_TEXT("incompatible shape"));
        } else {
            mp_binary_set_val_array(target->typecode, target->data, offset, value);
        }
    }
}

static uumpy_obj_ndarray_t *ndarray_from_list_tuple(mp_obj_t value, char typecode) {
    mp_int_t dim_count;
    mp_int_t dim_lengths[UUMPY_MAX_DIMS];

    dim_count = 0;
    mp_obj_t o = value;

    mp_int_t length;
    mp_obj_t *items;

    while (o && uumpy_util_get_list_tuple(o, &length, &items)) {
        if (dim_count >= UUMPY_MAX_DIMS) {
            mp_raise_ValueError(MP_ERROR_TEXT("too many dimensions"));
        }
        if (length) {
            dim_lengths[dim_count] = length;
            o = items[0];
            dim_count++;
        } else {
            o = NULL;
        }
    }

    uumpy_obj_ndarray_t *new_array = ndarray_new(typecode, dim_count, dim_lengths);

    ndarray_copy_list_tuple(value, new_array, 0, 0);

    return new_array;
}


static mp_obj_t ndarray_make_new(const mp_obj_type_t *type_in, mp_int_t n_args, mp_int_t n_kw, const mp_obj_t *args) {
    (void)type_in;
    mp_arg_check_num(n_args, n_kw, 1, 2, false);

    // Default type
    char typecode = UUMPY_DEFAULT_TYPE;

    if (n_args == 2) {
        size_t type_len;
        const char *typecode_ptr = mp_obj_str_get_data(args[1], &type_len);
        if (type_len != 1) {
            mp_raise_ValueError(MP_ERROR_TEXT("Data type should be a single character code"));
        }
        typecode = typecode_ptr[0];
    }

    // This also raises an exception if it's a bad typecode
    (void) mp_binary_get_size('@', typecode, NULL);

    mp_int_t dim_lengths[UUMPY_MAX_DIMS];
    mp_int_t n_dims = 0;
    mp_obj_t iterable = mp_getiter(args[0], NULL);
    mp_obj_t item;
    while ((item = mp_iternext(iterable)) != MP_OBJ_STOP_ITERATION) {
        if (n_dims >= UUMPY_MAX_DIMS) {
            mp_raise_ValueError(MP_ERROR_TEXT("too many dimensions"));
        }
        if (mp_obj_is_small_int(item)) {
            dim_lengths[n_dims] = MP_OBJ_SMALL_INT_VALUE(item);
            // DEBUG_printf("Dim %d has length %d\n", n_dims, dim_lengths[n_dims]);
        } else {
            mp_raise_ValueError(MP_ERROR_TEXT("Dimension sizes must be integers"));
        }
        n_dims++;
    }

    uumpy_obj_ndarray_t *array = ndarray_new(typecode, n_dims, dim_lengths);
    return MP_OBJ_FROM_PTR(array);
}

uumpy_obj_ndarray_t *uumpy_array_from_value(const mp_obj_t value, char typecode) {
    uumpy_obj_ndarray_t *new_array;

    if (mp_obj_is_type(value, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        new_array = ndarray_new_from_ndarray(value, typecode);
    } else if (mp_obj_is_type(value, MP_OBJ_FROM_PTR(&mp_type_list)) ||
               mp_obj_is_type(value, MP_OBJ_FROM_PTR(&mp_type_tuple)) ) {
        // DEBUG_printf("Making array from list\n");
        new_array = ndarray_from_list_tuple(value, typecode);
    } else {
        mp_obj_t value_len = mp_obj_len_maybe(value);

        if ((value_len != MP_OBJ_NULL) && !mp_obj_is_str(value)) {
            // DEBUG_printf("Making new 1d array from iterable\n");
            new_array = ndarray_new_1d_from_iterable(value, mp_obj_get_int(value_len), typecode);
        } else {
            // DEBUG_printf("Making array from single value\n");
            new_array = ndarray_new_0d(value, typecode);
        }
    }

    return new_array;
}

static mp_obj_t uumpy_array(size_t n_args, const mp_obj_t *args) {
    char typecode = UUMPY_DEFAULT_TYPE;

    if (n_args == 2) {
        typecode = *mp_obj_str_get_str(args[1]);
    }

    return MP_OBJ_FROM_PTR(uumpy_array_from_value(args[0], typecode));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(uumpy_array_obj, 1, 2, uumpy_array);

bool ndarray_compare_dimensions_counted(uumpy_obj_ndarray_t *left_in, uumpy_obj_ndarray_t *right_in, mp_int_t count) {
    for(mp_int_t i=0; i < count; i++) {
        if (left_in->dim_info[i].length != right_in->dim_info[i].length) {
            return false;
        }
    }

    return true;
}

bool ndarray_compare_dimensions(uumpy_obj_ndarray_t *left_in, uumpy_obj_ndarray_t *right_in) {
    if (left_in->dim_count != right_in->dim_count) {
        return false;
    }

    return ndarray_compare_dimensions_counted(left_in, right_in, left_in->dim_count);
}

// Returns true if the left array needed to be expanded
// If the entries are different lengths we pad the _start_ to that the ends align.
// If one dimension las length L>1 and the other has length 1 then reset to length L with stride 1
bool ndarray_broadcast(uumpy_obj_ndarray_t *left_in, uumpy_obj_ndarray_t *right_in,
                       uumpy_obj_ndarray_t **left_out, uumpy_obj_ndarray_t **right_out) {
    mp_int_t output_dim_count = MAX(left_in->dim_count, right_in->dim_count);
    bool left_touched = (output_dim_count != left_in->dim_count);
    uumpy_dim_info left_dim_info[UUMPY_MAX_DIMS];
    uumpy_dim_info right_dim_info[UUMPY_MAX_DIMS];

    // DEBUG_printf("Broadcasting between %d-D array and %d-D array into %d-D array\n",
    //              left_in->dim_count, right_in->dim_count, output_dim_count);

    // Dim counts are unsigned but we need a signed result
    mp_int_t l_index = ((mp_int_t) left_in->dim_count) - ((mp_int_t) output_dim_count);
    mp_int_t r_index = ((mp_int_t) right_in->dim_count) - ((mp_int_t) output_dim_count);

    for (mp_int_t i=0; i<output_dim_count; i++) {
        // DEBUG_printf("Output dim %d from left %d, right %d... ", i, l_index, r_index);
        if (l_index < 0) {
            // DEBUG_printf("too early for left, using right len=%d\n", right_in->dim_info[r_index].length);
            left_dim_info[i].length = right_in->dim_info[r_index].length;
            left_dim_info[i].stride = 0;
            right_dim_info[i] = right_in->dim_info[r_index];
            left_touched = true;
        } else if (r_index < 0) {
            // DEBUG_printf("too early for right, using left len=%d\n", left_in->dim_info[l_index].length);
            right_dim_info[i].length = left_in->dim_info[l_index].length;
            right_dim_info[i].stride = 0;
            left_dim_info[i] = left_in->dim_info[l_index];
        } else if (left_in->dim_info[l_index].length == right_in->dim_info[r_index].length) {
            // DEBUG_printf("equal lengths (%d)\n", left_in->dim_info[l_index].length);
            left_dim_info[i] = left_in->dim_info[l_index];
            right_dim_info[i] = right_in->dim_info[r_index];
        } else if (left_in->dim_info[l_index].length == 1) {
            // DEBUG_printf("left length is 1, right is %d\n", right_in->dim_info[r_index].length);
            left_dim_info[i].length = right_in->dim_info[r_index].length;
            left_dim_info[i].stride = 0;
            right_dim_info[i] = right_in->dim_info[r_index];
            left_touched = true;
        } else if (right_in->dim_info[r_index].length == 1) {
            // DEBUG_printf("right length is 1, left is %d\n", left_in->dim_info[l_index].length);
            right_dim_info[i].length = left_in->dim_info[l_index].length;
            right_dim_info[i].stride = 0;
            left_dim_info[i] = left_in->dim_info[l_index];
        } else {
            mp_raise_ValueError(MP_ERROR_TEXT("operands could not be broadcast together"));
        }

        l_index++;
        r_index++;

        // DEBUG_printf("  l.length=%d, l.stride=%d, r.length=%d, r.stride=%d\n",
        //              left_dim_info[i].length, left_dim_info[i].stride,
        //              right_dim_info[i].length, right_dim_info[i].stride);
    }



    // Make the new views
    *left_out = ndarray_new_view(left_in, left_in->base_offset,
                                 output_dim_count, left_dim_info);
    *right_out = ndarray_new_view(right_in, right_in->base_offset,
                                 output_dim_count, right_dim_info);

    // DEBUG_printf("Broadcast complete. Left%s touched\n", left_touched ? "" : " not");

    return left_touched;
}

static mp_obj_t ndarray_unary_op(mp_unary_op_t op, mp_obj_t o_in) {
    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(o_in);
    char result_typecode = 0;
    uumpy_universal_spec spec;

    switch (op) {
    case MP_UNARY_OP_LEN:
        if (o->dim_count == 0) {
            mp_raise_TypeError(MP_ERROR_TEXT("len() of unsized object"));
        } else {
            return MP_OBJ_NEW_SMALL_INT(o->dim_info[0].length);
        }
        break;

    case MP_UNARY_OP_BOOL:
        mp_raise_ValueError(MP_ERROR_TEXT("ambiguous; use any() or all()"));
        break;

    case MP_UNARY_OP_POSITIVE:
    case MP_UNARY_OP_NEGATIVE:
    case MP_UNARY_OP_ABS:
        ufunc_find_unary_op_spec(o, &result_typecode, op, &spec);
        break;

    default:
        return MP_OBJ_NULL;
    }

    uumpy_obj_ndarray_t *result = ndarray_new_shaped_like(result_typecode, o, 0);

    if (!ufunc_apply_unary(result, o, &spec)) {
        return MP_OBJ_NULL;
    } else {
        return ndarray_get_obj_or_0d(result);
    }
}

static mp_obj_t ndarray_binary_op(mp_binary_op_t op, mp_obj_t lhs_in, mp_obj_t rhs_in) {
    uumpy_obj_ndarray_t *lhs;
    uumpy_obj_ndarray_t *rhs;

    char result_typecode;
    bool in_place = false;
    bool reverse = false;

    // DEBUG_printf("Binary op: %d, lhs=%p, rhs=%p\n", op, lhs_in, rhs_in);

    lhs = MP_OBJ_TO_PTR(lhs_in);
    if (!mp_obj_is_type(rhs_in, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        // DEBUG_printf("Making new array from input %p\n", rhs_in);
        rhs = uumpy_array_from_value(rhs_in, lhs->typecode);
    } else {
        // DEBUG_printf("Using existing rhs object\n");
        rhs = MP_OBJ_TO_PTR(rhs_in);
    }

    // DEBUG_printf("Using views: lhs=%p, rhs=%p\n", lhs, rhs);

    switch (op) {
    case MP_BINARY_OP_IS:
        return (lhs == rhs) ? mp_const_true : mp_const_false;

    case MP_BINARY_OP_IN:
        return MP_OBJ_NULL;

    case MP_BINARY_OP_REVERSE_MAT_MULTIPLY:
        return ndarray_get_obj_or_0d(ndarray_dot_impl(rhs, lhs));
    case MP_BINARY_OP_MAT_MULTIPLY:
        return ndarray_get_obj_or_0d(ndarray_dot_impl(lhs, rhs));

    case MP_BINARY_OP_LESS:
    case MP_BINARY_OP_MORE:
    case MP_BINARY_OP_EQUAL:
    case MP_BINARY_OP_LESS_EQUAL:
    case MP_BINARY_OP_MORE_EQUAL:
    case MP_BINARY_OP_NOT_EQUAL:
        break;

    case MP_BINARY_OP_INPLACE_OR:
    case MP_BINARY_OP_INPLACE_XOR:
    case MP_BINARY_OP_INPLACE_AND:
    case MP_BINARY_OP_INPLACE_LSHIFT:
    case MP_BINARY_OP_INPLACE_RSHIFT:
    case MP_BINARY_OP_INPLACE_ADD:
    case MP_BINARY_OP_INPLACE_SUBTRACT:
    case MP_BINARY_OP_INPLACE_MULTIPLY:
    case MP_BINARY_OP_INPLACE_FLOOR_DIVIDE:
    case MP_BINARY_OP_INPLACE_TRUE_DIVIDE:
    case MP_BINARY_OP_INPLACE_MODULO:
    case MP_BINARY_OP_INPLACE_POWER:
        in_place = true;
        // Fall through
    case MP_BINARY_OP_OR:
    case MP_BINARY_OP_XOR:
    case MP_BINARY_OP_AND:
    case MP_BINARY_OP_LSHIFT:
    case MP_BINARY_OP_RSHIFT:
    case MP_BINARY_OP_ADD:
    case MP_BINARY_OP_SUBTRACT:
    case MP_BINARY_OP_MULTIPLY:
    case MP_BINARY_OP_FLOOR_DIVIDE:
    case MP_BINARY_OP_TRUE_DIVIDE:
    case MP_BINARY_OP_MODULO:
    case MP_BINARY_OP_POWER:
        break;

    case MP_BINARY_OP_REVERSE_OR:
    case MP_BINARY_OP_REVERSE_XOR:
    case MP_BINARY_OP_REVERSE_AND:
    case MP_BINARY_OP_REVERSE_LSHIFT:
    case MP_BINARY_OP_REVERSE_RSHIFT:
    case MP_BINARY_OP_REVERSE_ADD:
    case MP_BINARY_OP_REVERSE_SUBTRACT:
    case MP_BINARY_OP_REVERSE_MULTIPLY:
    case MP_BINARY_OP_REVERSE_FLOOR_DIVIDE:
    case MP_BINARY_OP_REVERSE_TRUE_DIVIDE:
    case MP_BINARY_OP_REVERSE_MODULO:
    case MP_BINARY_OP_REVERSE_POWER:
        op = op - MP_BINARY_OP_REVERSE_OR + MP_BINARY_OP_OR;
        reverse = true;
        break;

    default:
        return MP_OBJ_NULL;
    }

    if (in_place) {
        op = op - MP_BINARY_OP_INPLACE_OR + MP_BINARY_OP_OR;
        result_typecode = lhs->typecode;
    } else {
        result_typecode = 0;
    }

    uumpy_obj_ndarray_t *lhs_view;
    uumpy_obj_ndarray_t *rhs_view;

    if (ndarray_compare_dimensions(lhs, rhs)) {
        lhs_view = lhs;
        rhs_view = rhs;
    } else {
        // DEBUG_printf("Dimensions not the same. Broadcasting\n");

        bool left_expand;
        left_expand = ndarray_broadcast(lhs, rhs, &lhs_view, &rhs_view);
        if (left_expand && in_place) {
            mp_raise_ValueError(MP_ERROR_TEXT("non-broadcastable output operand"));
        }
    }

    if (reverse) {
        // DEBUG_printf("Reversing operands\n");
        uumpy_obj_ndarray_t *temp = lhs_view;
        lhs_view = rhs_view;
        rhs_view = temp;
    }

    // DEBUG_printf("Using views: lhs=%p, rhs=%p\n", lhs_view, rhs_view);

    uumpy_universal_spec spec;
    // Find the function spec, and result type if not in-place
    ufunc_find_binary_op_spec(lhs_view, rhs_view, &result_typecode, op, &spec);

    uumpy_obj_ndarray_t *result;

    if (!in_place) {
        // DEBUG_printf("Creating destination array\n");
        result = ndarray_new_shaped_like(result_typecode, lhs_view, 0);
    } else {
        // DEBUG_printf("In place. Using left side as destination array\n");
        result = lhs;
    }

    // DEBUG_printf("Calling application function\n");

    if (!ufunc_apply_binary(result, lhs_view, rhs_view, &spec)) {
        return MP_OBJ_NULL;
    } else {
        return ndarray_get_obj_or_0d(result);
    }
}

// We currently support subscripts that are int, slice, ellipsis or a tuple of these
static mp_obj_t ndarray_subscr(mp_obj_t self_in, mp_obj_t index_in, mp_obj_t value) {
    if (value == MP_OBJ_NULL) {
        return MP_OBJ_NULL; // Delete is not supported
    }

    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(self_in);

    mp_int_t subscript_count;
    mp_obj_t *dim_subscripts;

    if (!uumpy_util_get_list_tuple(index_in, &subscript_count, &dim_subscripts)) {
        subscript_count = 1;
        dim_subscripts = &index_in;
    }

    mp_int_t ellipsis_offset = -1;
    mp_int_t subs_offset;
    for(subs_offset=0; subs_offset < subscript_count; subs_offset++) {
        if (dim_subscripts[subs_offset] == MP_OBJ_FROM_PTR(&mp_const_ellipsis_obj)) {
            if (ellipsis_offset == -1) {
                ellipsis_offset = subs_offset;
            } else {
                mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("no more than one ellipsis allowed"));
            }
        }
    }

    // As we unpack the subscripts we have to track three separate offset:
    // how far through the list of subscripts we are, which dimension in the
    // source will get unpacked next and which dimension in the destination
    // will get filled next.

    mp_int_t slice_dim_offset = 0;
    mp_int_t target_dim_offset = 0;
    uumpy_dim_info target_dim_info[UUMPY_MAX_DIMS];
    mp_int_t target_base_offset = o->base_offset;

    for(subs_offset=0; subs_offset < subscript_count; subs_offset++) {
        if (slice_dim_offset >= o->dim_count) {
            mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("too many indices for source array"));
        }
        if (target_dim_offset >= UUMPY_MAX_DIMS) {
            mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("too many output dimensions"));
        }

        mp_obj_t item = dim_subscripts[subs_offset];
        if (item == mp_const_none) {
            // newaxis is an alias for None.
            // Add an output axis of length one. Does not consume a source axis
            target_dim_info[target_dim_offset].length = 1;
            target_dim_info[target_dim_offset].stride = 1;
            target_dim_offset++;
        } else if (item == MP_OBJ_FROM_PTR(&mp_const_ellipsis_obj)) {
            // Ellipsis.
            // Copy slice_dims to align end of subscripts with end of source
            // DEBUG_printf("Ellipsis at subscript %d out of %d\n", subs_offset, subscript_count);
            // DEBUG_printf("Total source dimensions: %d\n", o->dim_count);
            // DEBUG_printf("Current source dimension: %d\n", slice_dim_offset);
            mp_int_t copy_up_to = o->dim_count - (subscript_count - (subs_offset + 1));

            // DEBUG_printf("Copying source up to %d\n", copy_up_to);

            while (slice_dim_offset < copy_up_to) {
                // DEBUG_printf("  copying %d to %d\n", slice_dim_offset, target_dim_offset);
                if (target_dim_offset >= UUMPY_MAX_DIMS) {
                    mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("too many output dimensions"));
                }
                target_dim_info[target_dim_offset] = o->dim_info[slice_dim_offset];
                slice_dim_offset++;
                target_dim_offset++;
            }
        } else if (mp_obj_is_type(item, &mp_type_slice)) {
            // Slice
            // Resolve slice, multiply stride, adjust target_base_offset
            mp_bound_slice_t slice_info;
            mp_obj_slice_indices(item, o->dim_info[slice_dim_offset].length, &slice_info);
            // Divide length by step, rounding UP
            mp_int_t slice_count;

            if (slice_info.step > 0) {
                slice_count = (slice_info.stop + (slice_info.step - 1) - slice_info.start) / slice_info.step;
            } else {
                slice_count = (slice_info.start - (slice_info.step + 1) - slice_info.stop) / (-slice_info.step);
            }
            // DEBUG_printf("Slice: start %d, stop %d, step %d, count %d\n",
            //              slice_info.start, slice_info.stop, slice_info.step, slice_count);

            target_base_offset += (o->dim_info[slice_dim_offset].stride * slice_info.start);
            target_dim_info[target_dim_offset].length = slice_count;
            target_dim_info[target_dim_offset].stride = o->dim_info[slice_dim_offset].stride * slice_info.step;

            // Consume one source dim and one target dim
            slice_dim_offset++;
            target_dim_offset++;
        } else if (mp_obj_is_small_int(item)) {
            // Simpler integer
            // Adjust base offset, consume source dim but not target
            mp_int_t index = MP_OBJ_SMALL_INT_VALUE(item);

            if (index < 0) {
                index += o->dim_info[slice_dim_offset].length;
            }
            if (index < 0 || index >= o->dim_info[slice_dim_offset].length) {
                mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("index out of range"));
            }
            target_base_offset += (index * o->dim_info[slice_dim_offset].stride);
            slice_dim_offset++;
        } else {
            // Something else
            mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("unsupported index type"));
        }
    }
    // If we didn't use all the source dimensions then just copy them over
    while (slice_dim_offset < o->dim_count) {
        if (target_dim_offset >= UUMPY_MAX_DIMS) {
            mp_raise_msg(&mp_type_IndexError, MP_ERROR_TEXT("too many output dimensions"));
        }
        target_dim_info[target_dim_offset] = o->dim_info[slice_dim_offset];
        slice_dim_offset++;
        target_dim_offset++;
    }

    if (value == MP_OBJ_SENTINEL) {
        if (target_dim_offset == 0) {
            return mp_binary_get_val_array(o->typecode, o->data, target_base_offset);
        } else {
            return MP_OBJ_FROM_PTR(ndarray_new_view(o, target_base_offset,
                                                    target_dim_offset, target_dim_info));
        }
    } else {
        if (target_dim_offset == 0) {
            mp_binary_set_val_array(o->typecode, o->data, target_base_offset, value);
            return mp_const_none;
        } else {
            uumpy_obj_ndarray_t *dest, *src;
            dest = ndarray_new_view(o, target_base_offset, target_dim_offset, target_dim_info);

            if (!mp_obj_is_type(value, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
                src = uumpy_array_from_value(value, o->typecode);
            } else {
                src = MP_OBJ_TO_PTR(value);
            }

            if (!ndarray_compare_dimensions(src, dest)) {
                // Try broadcasting
                if (ndarray_broadcast(dest, src, &dest, &src)) {
                    mp_raise_ValueError(MP_ERROR_TEXT("value can not be broadcast into slice"));
                }
            }
            // Copy values
            uumpy_universal_spec copy_spec;

            ufunc_find_copy_spec(src, dest, NULL, &copy_spec);

            ufunc_apply_unary(dest, src, &copy_spec);

            return mp_const_none;
        }
    }
    return MP_OBJ_NULL;
}

static mp_obj_t ndarray_iterator_new(mp_obj_t array_in, mp_obj_iter_buf_t *iter_buf) {
    return MP_OBJ_NULL;
}

static mp_obj_t ndarray_shape(mp_obj_t self_in) {
    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(self_in);
    mp_obj_t values[UUMPY_MAX_DIMS];

    for(mp_int_t i = 0; i < o->dim_count; i++) {
        values[i] = MP_OBJ_NEW_SMALL_INT(o->dim_info[i].length);
    }

    return mp_obj_new_tuple(o->dim_count, values);
}
static MP_DEFINE_CONST_FUN_OBJ_1(ndarray_shape_obj, ndarray_shape);

static mp_obj_t ndarray_transpose(size_t n_args, const mp_obj_t *args) {
    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(args[0]);
    uumpy_dim_info new_dim_info[UUMPY_MAX_DIMS];
    mp_int_t new_order_indices[UUMPY_MAX_DIMS];
    mp_int_t dim_count = o->dim_count;

    if (n_args == 1) {
        for (mp_int_t i=0; i < dim_count; i++) {
            new_order_indices[i] = (dim_count - 1) - i;
        }
    } else {
        mp_int_t length;
        mp_obj_t *items;

        if (uumpy_util_get_list_tuple(args[1], &length, &items)) {
            if (length != dim_count) {
                mp_raise_ValueError(MP_ERROR_TEXT("axes don't match array"));
            }
            // NOTE: This relies on UUMPY_MAX_DIMS being less than the word size
            mp_int_t dim_bits = (1 << dim_count) - 1;

            for (mp_int_t i=0; i < dim_count; i++) {
                mp_int_t d = mp_obj_get_int(items[i]);
                if (dim_bits & (1 << d)) {
                    new_order_indices[i] = d;
                    dim_bits &= ~(1 << d);
                } else {
                    // This lumps together repeated and out of range dimensions
                    mp_raise_ValueError(MP_ERROR_TEXT("invalid transpose dimension"));
                }
            }
        } else {
            mp_raise_ValueError(MP_ERROR_TEXT("transpose order must be a list or tuple"));
        }
    }

    for (mp_int_t i=0; i < dim_count; i++) {
        new_dim_info[i] = o->dim_info[new_order_indices[i]];
    }

    uumpy_obj_ndarray_t *result = ndarray_new_view(o, o->base_offset, o->dim_count, new_dim_info);

    return MP_OBJ_FROM_PTR(result);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(ndarray_transpose_obj, 1, 2, ndarray_transpose);

static mp_obj_t ndarray_reshape(mp_obj_t self_in, mp_obj_t new_shape_obj) {
    uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(self_in);
    uumpy_obj_ndarray_t *new_array = NULL;

    if (!o->simple) {
        o = ndarray_new_from_ndarray(self_in, 0);
    }

    mp_int_t original_count = 1;
    for (mp_int_t i=0; i < o->dim_count; i++) {
        original_count *= o->dim_info[i].length;
    }

    uumpy_dim_info dim_info[UUMPY_MAX_DIMS];
    mp_int_t dim_count;
    mp_obj_t *values;

    if (uumpy_util_get_list_tuple(new_shape_obj, &dim_count, &values)) {
        mp_int_t stride = 1;
        if (dim_count > UUMPY_MAX_DIMS) {
            mp_raise_ValueError(MP_ERROR_TEXT("too many dimensions"));
        }

        for (mp_int_t i = dim_count-1; i >= 0; i--) {
            dim_info[i].length = mp_obj_get_int(values[i]);
            dim_info[i].stride = stride;
            stride *= dim_info[i].length;
        }

        if (stride != original_count) {
            mp_raise_ValueError(MP_ERROR_TEXT("new shape has different size"));
        }

        new_array = ndarray_new_view(o, 0, dim_count, dim_info);
    } else {
        mp_raise_ValueError(MP_ERROR_TEXT("new shape must be a list or tuple"));
    }

    return MP_OBJ_FROM_PTR(new_array);
}
static MP_DEFINE_CONST_FUN_OBJ_2(ndarray_reshape_obj, ndarray_reshape);

typedef struct _uumpy_isclose_spec {
    mp_float_t rtol;
    mp_float_t atol;
    bool equal_nan;
} uumpy_isclose_spec;

static bool _uumpy_isclose_test(mp_float_t a, mp_float_t b, uumpy_isclose_spec *spec) {
    if (isnan(a) && !isnan(b)) {
        return spec->equal_nan;
    }

    if (isnan(a) || isnan(b)) {
        return false;
    }

    if (a == b) {
        return true;
    }

    mp_float_t diff = MICROPY_FLOAT_C_FUN(fabs)(a - b);
    b = MICROPY_FLOAT_C_FUN(fabs)(b);

    return (diff <= (spec->atol + spec->rtol  * b));
}

static bool _uumpy_isclose_func_float(size_t depth,
                                       uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                       uumpy_obj_ndarray_t *src1, mp_int_t src1_offset,
                                       uumpy_obj_ndarray_t *src2, mp_int_t src2_offset,
                                       struct _uumpy_universal_spec *spec) {
    (void) depth;

    mp_float_t *src1_data = src1->data;
    mp_float_t *src2_data = src2->data;
    unsigned char *dest_data = dest->data;

    dest_data[dest_offset] = _uumpy_isclose_test(src1_data[src1_offset],
                                                 src2_data[src2_offset],
                                                 spec->context);
    return true;
}

static bool _uumpy_isclose_func_fallback(size_t depth,
                                          uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                          uumpy_obj_ndarray_t *src1, mp_int_t src1_offset,
                                          uumpy_obj_ndarray_t *src2, mp_int_t src2_offset,
                                          struct _uumpy_universal_spec *spec) {
    (void) depth;
    unsigned char *dest_data = dest->data;

    mp_obj_t a_obj = mp_binary_get_val_array(src1->typecode, src1->data, src1_offset);
    mp_obj_t b_obj = mp_binary_get_val_array(src2->typecode, src2->data, src2_offset);

    mp_float_t a = mp_obj_get_float(a_obj);
    mp_float_t b = mp_obj_get_float(b_obj);

    dest_data[dest_offset] = _uumpy_isclose_test(a, b, spec->context);

    return true;
}


static mp_obj_t uumpy_isclose(mp_uint_t n_args, const mp_obj_t *pos_args,
                              mp_map_t *kw_args) {
    enum {
        ARG_a,
        ARG_b,
        ARG_rtol,
        ARG_atol,
        ARG_equal_nan,
    };

    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_a,         MP_ARG_REQUIRED | MP_ARG_OBJ,  {.u_obj = mp_const_none} },
        { MP_QSTR_b,         MP_ARG_REQUIRED | MP_ARG_OBJ,  {.u_obj = mp_const_none} },
        { MP_QSTR_rtol,      MP_ARG_KW_ONLY  | MP_ARG_OBJ,  {.u_obj = mp_const_none} },
        { MP_QSTR_atol,      MP_ARG_KW_ONLY  | MP_ARG_OBJ,  {.u_obj = mp_const_none} },
        { MP_QSTR_equal_nan, MP_ARG_KW_ONLY  | MP_ARG_BOOL, {.u_bool = false} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    uumpy_obj_ndarray_t *a, *b;
    uumpy_isclose_spec close_spec = {
        .rtol=1e-05,
        .atol=1e-08,
    };
    close_spec.equal_nan = args[ARG_equal_nan].u_bool;

    if (!mp_obj_is_type(args[ARG_a].u_obj, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        a = uumpy_array_from_value(args[ARG_a].u_obj, UUMPY_DEFAULT_TYPE);
    } else {
        a = MP_OBJ_TO_PTR(args[ARG_a].u_obj);
    }

    if (!mp_obj_is_type(args[ARG_b].u_obj, MP_OBJ_FROM_PTR(&uumpy_type_ndarray))) {
        b = uumpy_array_from_value(args[ARG_b].u_obj, UUMPY_DEFAULT_TYPE);
    } else {
        b = MP_OBJ_TO_PTR(args[ARG_b].u_obj);
    }

    if (args[ARG_rtol].u_obj != mp_const_none) {
        close_spec.rtol = mp_obj_get_float(args[ARG_rtol].u_obj);
    }
    if (args[ARG_atol].u_obj != mp_const_none) {
        close_spec.atol = mp_obj_get_float(args[ARG_atol].u_obj);
    }

    if (!ndarray_compare_dimensions(a, b)) {
        ndarray_broadcast(a, b, &a, &b);
    }

    uumpy_obj_ndarray_t *result = ndarray_new_shaped_like('B', a, 0);
    uumpy_universal_spec spec = {
        .layers = 0,
    };
    spec.context = &close_spec;

    if (a->typecode == UUMPY_DEFAULT_TYPE && b->typecode == UUMPY_DEFAULT_TYPE) {
        spec.apply_fn.binary = _uumpy_isclose_func_float;
    } else {
        spec.apply_fn.binary = _uumpy_isclose_func_fallback;
    }

    ufunc_apply_binary(result, a, b, &spec);

    return MP_OBJ_FROM_PTR(result);
}
MP_DEFINE_CONST_FUN_OBJ_KW(uumpy_isclose_obj, 1, uumpy_isclose);

static const mp_rom_map_elem_t ndarray_locals_dict_table[] = {
    //    { MP_ROM_QSTR(MP_QSTR_T), MP_ROM_PTR(&ndarray_T_property_obj) },
    //    { MP_ROM_QSTR(MP_QSTR_shape), MP_ROM_PTR(&array_shape_property_obj) },
};

static MP_DEFINE_CONST_DICT(ndarray_locals_dict, ndarray_locals_dict_table);


static void ndarray_attr(mp_obj_t self_in, qstr attr, mp_obj_t *dest) {
    if (dest[0] != MP_OBJ_NULL) {
        // not load attribute
        return;
    }
    // uumpy_obj_ndarray_t *o = MP_OBJ_TO_PTR(self_in);

    if (attr == MP_QSTR_T) {
        dest[0] = ndarray_transpose(1, &self_in);
    } else if (attr == MP_QSTR_shape) {
        dest[0] = ndarray_shape(self_in);
    } else if (attr == MP_QSTR_reshape) {
        dest[0] = MP_OBJ_FROM_PTR(&ndarray_reshape_obj);
        dest[1] = self_in;
    } else if (attr == MP_QSTR_transpose) {
        dest[0] = MP_OBJ_FROM_PTR(&ndarray_transpose_obj);
        dest[1] = self_in;
    } else if (attr == MP_QSTR_dot) {
        dest[0] = MP_OBJ_FROM_PTR(&ndarray_dot_obj);
        dest[1] = self_in;
    }

}


MP_DEFINE_CONST_OBJ_TYPE(
        uumpy_type_ndarray,
        MP_QSTR_ndarray,
        MP_TYPE_FLAG_NONE,
        print, ndarray_print,
        make_new, ndarray_make_new,
        locals_dict, &ndarray_locals_dict,
        unary_op, ndarray_unary_op,
        binary_op, ndarray_binary_op,
        subscr, ndarray_subscr,
        iter, ndarray_iterator_new,
        attr, ndarray_attr
);

// Define all properties of the uumpy module.
// Table entries are key/value pairs of the attribute name (a string)
// and the MicroPython object reference.
// All identifiers and strings are written as MP_QSTR_xxx and will be
// optimized to word-sized integers by the build system (interned strings).
static const mp_rom_map_elem_t uumpy_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_uumpy) },
    { MP_ROM_QSTR(MP_QSTR_ndarray), MP_ROM_PTR(&uumpy_type_ndarray) },
    { MP_ROM_QSTR(MP_QSTR_newaxis), MP_ROM_NONE },
    { MP_ROM_QSTR(MP_QSTR_shape), MP_ROM_PTR(&ndarray_shape_obj) },
    { MP_ROM_QSTR(MP_QSTR_reshape), MP_ROM_PTR(&ndarray_reshape_obj) },
    { MP_ROM_QSTR(MP_QSTR_array), MP_ROM_PTR(&uumpy_array_obj) },
    { MP_ROM_QSTR(MP_QSTR_transpose), MP_ROM_PTR(&ndarray_transpose_obj) },
    { MP_ROM_QSTR(MP_QSTR_dot), MP_ROM_PTR(&ndarray_dot_obj) },

    { MP_ROM_QSTR(MP_QSTR_sin), MP_ROM_PTR(&uumpy_math_sin_obj) },
    { MP_ROM_QSTR(MP_QSTR_cos), MP_ROM_PTR(&uumpy_math_cos_obj) },
    { MP_ROM_QSTR(MP_QSTR_tan), MP_ROM_PTR(&uumpy_math_tan_obj) },
    { MP_ROM_QSTR(MP_QSTR_asin), MP_ROM_PTR(&uumpy_math_asin_obj) },
    { MP_ROM_QSTR(MP_QSTR_acos), MP_ROM_PTR(&uumpy_math_acos_obj) },
    { MP_ROM_QSTR(MP_QSTR_atan), MP_ROM_PTR(&uumpy_math_atan_obj) },

#if UUMPY_ENABLE_HYPERBOLIC
    { MP_ROM_QSTR(MP_QSTR_sinh), MP_ROM_PTR(&uumpy_math_sinh_obj) },
    { MP_ROM_QSTR(MP_QSTR_cosh), MP_ROM_PTR(&uumpy_math_cosh_obj) },
    { MP_ROM_QSTR(MP_QSTR_tanh), MP_ROM_PTR(&uumpy_math_tanh_obj) },
    { MP_ROM_QSTR(MP_QSTR_asinh), MP_ROM_PTR(&uumpy_math_asinh_obj) },
    { MP_ROM_QSTR(MP_QSTR_acosh), MP_ROM_PTR(&uumpy_math_acosh_obj) },
    { MP_ROM_QSTR(MP_QSTR_atanh), MP_ROM_PTR(&uumpy_math_atanh_obj) },
#endif

#if UUMPY_ENABLE_LINALG
    { MP_ROM_QSTR(MP_QSTR_linalg), MP_ROM_PTR(&uumpy_linalg_module) },
#endif

    { MP_ROM_QSTR(MP_QSTR_log), MP_ROM_PTR(&uumpy_math_log_obj) },
    { MP_ROM_QSTR(MP_QSTR_exp), MP_ROM_PTR(&uumpy_math_exp_obj) },

    { MP_ROM_QSTR(MP_QSTR_isclose), MP_ROM_PTR(&uumpy_isclose_obj) },

    { MP_ROM_QSTR(MP_QSTR_LinAlgError), MP_ROM_PTR(&uumpy_linalg_type_LinAlgError) },

    { MP_ROM_QSTR(MP_QSTR_max), MP_ROM_PTR(&uumpy_reduction_max_obj) },
    { MP_ROM_QSTR(MP_QSTR_min), MP_ROM_PTR(&uumpy_reduction_min_obj) },
    { MP_ROM_QSTR(MP_QSTR_sum), MP_ROM_PTR(&uumpy_reduction_sum_obj) },
    { MP_ROM_QSTR(MP_QSTR_prod), MP_ROM_PTR(&uumpy_reduction_prod_obj) },
    { MP_ROM_QSTR(MP_QSTR_average), MP_ROM_PTR(&uumpy_reduction_average_obj) },
    { MP_ROM_QSTR(MP_QSTR_any), MP_ROM_PTR(&uumpy_reduction_any_obj) },
    { MP_ROM_QSTR(MP_QSTR_all), MP_ROM_PTR(&uumpy_reduction_all_obj) },


};
static MP_DEFINE_CONST_DICT(uumpy_module_globals, uumpy_module_globals_table);

// Define module object.
const mp_obj_module_t uumpy_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&uumpy_module_globals,
};

// Register the module to make it available in Python
MP_REGISTER_MODULE(MP_QSTR_uumpy, uumpy_user_cmodule);
