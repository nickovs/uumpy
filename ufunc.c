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

#include "py/runtime.h"
#include "py/builtin.h"
#include "py/binary.h"
#include "py/objtuple.h"

#include "moduumpy.h"
#include "ufunc.h"

// This non-recursive implementation looks more complex than the obvious
// recursive version but it both uses less stack and is faster.
bool ufunc_apply_binary(uumpy_obj_ndarray_t *dest,
                        uumpy_obj_ndarray_t *src1,
                        uumpy_obj_ndarray_t *src2,
                        struct _uumpy_universal_spec *spec) {
    size_t iterate_layers = dest->dim_count - spec->layers;

    size_t layer_indecies[UUMPY_MAX_DIMS];
    size_t dest_offset = dest->base_offset;
    size_t src1_offset = src1->base_offset;
    size_t src2_offset = src2->base_offset;
    bool result = true;
    mp_int_t l;

    // The layer_indecies count down, since it's quicker
    for (size_t i=0; i < iterate_layers; i++) {
        layer_indecies[i] = dest->dim_info[i].length;
    }

    do {
        result &= spec->apply_fn.binary(iterate_layers,
                                        dest, dest_offset,
                                        src1, src1_offset,
                                        src2, src2_offset,
                                        spec);
        for (l = iterate_layers-1; l >=0; l--) {
            src1_offset += src1->dim_info[l].stride;
            src2_offset += src2->dim_info[l].stride;
            dest_offset += dest->dim_info[l].stride;
            layer_indecies[l]--;

            if (layer_indecies[l] > 0) {
                break;
            } else {
                // Reset this row and allow moving on to the next one
                size_t ll = dest->dim_info[l].length;
                layer_indecies[l] = ll;
                src1_offset -= ll * src1->dim_info[l].stride;
                src2_offset -= ll * src2->dim_info[l].stride;
                dest_offset -= ll * dest->dim_info[l].stride;
            }
        }
    } while (l >= 0);

    return result;
}

bool ufunc_apply_unary(uumpy_obj_ndarray_t *dest,
                       uumpy_obj_ndarray_t *src,
                       uumpy_universal_spec *spec) {
    size_t iterate_layers = dest->dim_count - spec->layers;

    size_t layer_indecies[UUMPY_MAX_DIMS];
    size_t dest_offset = dest->base_offset;
    size_t src_offset = src->base_offset;
    bool result = true;
    mp_int_t l;

    // The layer_indecies count down, since it's quicker
    for (size_t i=0; i < iterate_layers; i++) {
        layer_indecies[i] = dest->dim_info[i].length;
    }

    do {
        result &= spec->apply_fn.unary(iterate_layers,
                                       dest, dest_offset,
                                       src, src_offset,
                                       spec);
        for (l = iterate_layers-1; l >=0; l--) {
            src_offset += src->dim_info[l].stride;
            dest_offset += dest->dim_info[l].stride;
            layer_indecies[l]--;

            if (layer_indecies[l] > 0) {
                break;
            } else {
                // Reset this row and allow moving on to the next one
                size_t ll = dest->dim_info[l].length;
                layer_indecies[l] = ll;
                src_offset -= ll * src->dim_info[l].stride;
                dest_offset -= ll * dest->dim_info[l].stride;
            }
        }
    } while (l >= 0);

    return result;
}


// This is a fall-back copy function that lets micropython deal with casting.
STATIC bool ufunc_copy_fallback(size_t depth,
                                uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                uumpy_obj_ndarray_t *src, size_t src_offset,
                                struct _uumpy_universal_spec *spec) {
    (void) depth;
    (void) spec;
    mp_obj_t value = mp_binary_get_val_array(src->typecode, src->data, src_offset);
    mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, value);
    return true;
}

STATIC bool ufunc_copy_same_type(size_t depth,
                                 uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                 uumpy_obj_ndarray_t *src, size_t src_offset,
                                 struct _uumpy_universal_spec *spec) {
    (void) depth;
    // DEBUG_printf("Copy same type at depth %d, from %d to %d, count=%d\n", depth, src_offset, dest_offset, spec->extra.c_count);

    size_t value_size = spec->value_size;
    memcpy(((char *) dest->data) + dest_offset * value_size, ((char *) src->data) + src_offset * value_size, spec->extra.c_count * value_size);
    return true;
}


// If there is no optimised code for any given operation, just let micropython do it
STATIC bool ufunc_universal_binary_op_fallback(size_t depth,
                                               uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                               uumpy_obj_ndarray_t *src1, size_t src1_offset,
                                               uumpy_obj_ndarray_t *src2, size_t src2_offset,
                                               struct _uumpy_universal_spec *spec) {
    (void) depth;

    // DEBUG_printf("Fallback src1=%p (%p + %d), src2=%p (%p+ %d)\n", src1, src1->data, src1_offset, src2, src1->data, src2_offset);

    mp_obj_t value1 = mp_binary_get_val_array(src1->typecode, src1->data, src1_offset);
    mp_obj_t value2 = mp_binary_get_val_array(src2->typecode, src2->data, src2_offset);

    // DEBUG_printf("Fallback binary op: %d lhs: %p rhs: %p\n", spec->extra.b_op, value1, value2);

    mp_obj_t result = mp_binary_op(spec->extra.b_op, value1, value2);
    if (result == MP_OBJ_NULL) {
        // DEBUG_printf(" FAILED\n");
        return false;
    } else {
        mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, result);
        // DEBUG_printf(" result: %p\n", result);
        return true;
    }
}

STATIC bool ufunc_universal_unary_op_fallback(size_t depth,
                                              uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                              uumpy_obj_ndarray_t *src, size_t src_offset,
                                              struct _uumpy_universal_spec *spec) {
    (void) depth;
    mp_obj_t value = mp_binary_get_val_array(src->typecode, src->data, src_offset);

    // DEBUG_printf("Fallback unary op: %d value: %p ", spec->extra.b_op, value);

    mp_obj_t result = mp_unary_op(spec->extra.u_op, value);
    if (result == MP_OBJ_NULL) {
        // DEBUG_printf(" FAILED\n");
        return false;
    } else {
        mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, result);
        // DEBUG_printf(" result: %p\n", result);
        return true;
    }
}

STATIC bool ufunc_unary_float_func_fallback(size_t depth,
                                            uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                            uumpy_obj_ndarray_t *src, size_t src_offset,
                                            struct _uumpy_universal_spec *spec) {
    (void) depth;
    mp_obj_t value = mp_binary_get_val_array(src->typecode, src->data, src_offset);
    mp_float_t x = mp_obj_get_float(value);
    mp_float_t ans = spec->extra.f_func(x);
    if ((isnan(ans) && !isnan(x)) || (isinf(ans) && !isinf(x))) {
        mp_raise_ValueError("math domain error");
    }

    mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, mp_obj_new_float(ans));

    return true;
}

// Applies a function to a whole line, assuming both are float arrays
STATIC bool ufunc_unary_float_func_floats_1d(size_t depth,
                                             uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                             uumpy_obj_ndarray_t *src, size_t src_offset,
                                             struct _uumpy_universal_spec *spec) {
    mp_float_t *src_ptr = (mp_float_t *) src->data;
    mp_float_t *dest_ptr = (mp_float_t *) dest->data;
    mp_int_t src_stride = src->dim_info[depth].stride;
    mp_int_t dest_stride = dest->dim_info[depth].stride;

    for (size_t i = dest->dim_info[depth].length; i > 0; i--) {
        mp_float_t x = (src_ptr[src_offset]);
        mp_float_t ans = spec->extra.f_func(x);

        if ((isnan(ans) && !isnan(x)) || (isinf(ans) && !isinf(x))) {
            mp_raise_ValueError("math domain error");
        }
                        dest_ptr[dest_offset] = ans;

        src_offset += src_stride;
        dest_offset += dest_stride;
    }

    return true;
}

void ufunc_mul_acc_fallback(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                            uumpy_obj_ndarray_t *src1, size_t src1_offset, size_t src1_dim,
                            uumpy_obj_ndarray_t *src2, size_t src2_offset, size_t src2_dim) {
    // DEBUG_printf("MAC fallbabck: dest offset=%d, src1 offset=%d, dim=%d, src2 offset=%d, dim=%d\n",
    //              dest_offset, src1_offset, src1_dim, src2_offset, src2_dim);
    size_t length = src1->dim_info[src1_dim].length;
    if (length != src2->dim_info[src2_dim].length) {
        // DEBUG_printf("src1 depth: %d, src2 depth: %d : %d != %d", src1_dim, src2_dim, length, src2->dim_info[src2_dim].length);
        mp_raise_ValueError("dimension mis-match");
    }

    mp_obj_t acc = (dest->typecode == UUMPY_DEFAULT_TYPE) ? mp_obj_new_float(0.0) : MP_OBJ_NEW_SMALL_INT(0);
    mp_obj_t prod, v1, v2;

    for (size_t i=0; i < src1->dim_info[src1_dim].length; i++) {
        v1 = mp_binary_get_val_array(src1->typecode, src1->data, src1_offset);
        v2 = mp_binary_get_val_array(src2->typecode, src2->data, src2_offset);

        prod = mp_binary_op(MP_BINARY_OP_MULTIPLY, v1, v2);
        if (prod == MP_OBJ_NULL) {
            mp_raise_ValueError("could not multiply components");
        }

        acc =  mp_binary_op(MP_BINARY_OP_ADD, prod, acc);
        if (acc == MP_OBJ_NULL) {
            mp_raise_ValueError("could not accumulate components");
        }

        src1_offset += src1->dim_info[src1_dim].stride;
        src2_offset += src2->dim_info[src2_dim].stride;
    }

    mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, acc);
}

static char type_expand(char lhs_type, char rhs_type) {
    // FIXME: This is not the right way to do this...
    return lhs_type;
}

// Find the 'best' universal spec for the operation given the types
void ufunc_find_binary_op_spec(uumpy_obj_ndarray_t *src1, uumpy_obj_ndarray_t *src2,
                               char *dest_type_in_out, mp_binary_op_t op,
                               uumpy_universal_spec *spec_out) {
    char result_type = *dest_type_in_out;
    if (result_type == 0) {
        switch (op) {
        case MP_BINARY_OP_LESS:
        case MP_BINARY_OP_MORE:
        case MP_BINARY_OP_EQUAL:
        case MP_BINARY_OP_LESS_EQUAL:
        case MP_BINARY_OP_MORE_EQUAL:
        case MP_BINARY_OP_NOT_EQUAL:
            result_type = 'B';
            break;

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
            result_type = type_expand(src1->typecode, src2->typecode);
            break;

        default:
            mp_raise_ValueError("Unsupported universal operator");
            break;
        }
        *dest_type_in_out = result_type;
    }

    uumpy_universal_spec spec = {
        .layers = 0,
        .apply_fn.binary = &ufunc_universal_binary_op_fallback,
        .extra.b_op = op,
    };

    *spec_out = spec;
}

void ufunc_find_unary_op_spec(uumpy_obj_ndarray_t *src,
                              char *dest_type_in_out, mp_unary_op_t op,
                              uumpy_universal_spec *spec_out) {
    char result_type = *dest_type_in_out;
    if (result_type == 0) {
        switch (op) {
        case MP_UNARY_OP_POSITIVE:
        case MP_UNARY_OP_NEGATIVE:
        case MP_UNARY_OP_ABS:
            result_type = src->typecode;
            break;

        default:
            mp_raise_ValueError("Unsupported universal operator");
            break;
        }
        *dest_type_in_out = result_type;
    }

    uumpy_universal_spec spec = {
        .layers = 0,
        .apply_fn.unary = &ufunc_universal_unary_op_fallback,
        .extra.u_op = op,
    };

    *spec_out = spec;
}

void ufunc_find_unary_float_func_spec(uumpy_obj_ndarray_t *src,
                                      char *dest_type_in_out, uumpy_unary_float_func f,
                                      uumpy_universal_spec *spec_out) {
    if (*dest_type_in_out == 0) {
        *dest_type_in_out = UUMPY_DEFAULT_TYPE;
    }

    if ((src->dim_count > 0) &&
        (src->typecode == UUMPY_DEFAULT_TYPE) &&
        (*dest_type_in_out == UUMPY_DEFAULT_TYPE)) {
        uumpy_universal_spec spec = {
            .layers = 1,
            .apply_fn.unary = &ufunc_unary_float_func_floats_1d,
            .extra.f_func = f,
        };

        *spec_out = spec;
    } else {
        uumpy_universal_spec spec = {
            .layers = 0,
            .apply_fn.unary = &ufunc_unary_float_func_fallback,
            .extra.f_func = f,
        };

        *spec_out = spec;
    }
}

void ufunc_find_copy_spec(uumpy_obj_ndarray_t *src,
                          uumpy_obj_ndarray_t *dest,
                          char *dest_type_out,
                          uumpy_universal_spec *spec_out) {
    char dest_type = dest ? dest->typecode : src->typecode;

    if (dest_type_out) {
        *dest_type_out = dest_type;
    }

    if (dest_type == src->typecode) {
        size_t chunk_size = 1;

        mp_int_t i = src->dim_count-1;

        while((i >= 0) &&
              (src->dim_info[i].stride == chunk_size) &&
              (!dest || dest->dim_info[i].stride == chunk_size) ) {
            chunk_size *= src->dim_info[i].length;
            i--;
        }

        uumpy_universal_spec copy_spec = {
            .layers = (src->dim_count-1) - i,
            .apply_fn.unary = &ufunc_copy_same_type,
            .extra.c_count = chunk_size,
        };

        copy_spec.value_size = mp_binary_get_size('@', dest_type, NULL),

        *spec_out = copy_spec;
    } else {
        uumpy_universal_spec copy_spec = {
            .layers = 0,
            .apply_fn.unary = &ufunc_copy_fallback,
        };

        *spec_out = copy_spec;
    }
}
