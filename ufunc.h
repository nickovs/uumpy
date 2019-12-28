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

struct _uumpy_universal_spec;

// A function that iterates across the last dimension to perform a function
typedef bool(*uumpy_universal_binary)(size_t depth,
                                      uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                      uumpy_obj_ndarray_t *src1, size_t src1_offset,
                                      uumpy_obj_ndarray_t *src2, size_t src2_offset,
                                      struct _uumpy_universal_spec *spec);
typedef bool(*uumpy_universal_unary)(size_t depth,
                                     uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                     uumpy_obj_ndarray_t *src, size_t src_offset,
                                     struct _uumpy_universal_spec *spec);

typedef void(*uumpy_multiply_accumulate)(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                         uumpy_obj_ndarray_t *src1, size_t src1_offset, size_t src1_dim,
                                         uumpy_obj_ndarray_t *src2, size_t src2_offset, size_t src2_dim);

typedef void(*uumpy_reduction_init)(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                    uumpy_obj_ndarray_t *src, size_t src_offset,
                                    struct _uumpy_universal_spec *spec, void *state);
typedef void(*uumpy_reduction_unary)(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                     uumpy_obj_ndarray_t *src, size_t src_offset,
                                     struct _uumpy_universal_spec *spec,
                                     void *state, bool is_first);
typedef void(*uumpy_reduction_finish)(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                                      struct _uumpy_universal_spec *spec, void *state);

typedef struct _uumpy_reduction_spec {
    size_t state_size;
    uumpy_reduction_init init_func;
    uumpy_reduction_unary iter_func;
    uumpy_reduction_finish finish_func;
} uumpy_reduction_spec;

typedef mp_float_t(*uumpy_unary_float_func)(mp_float_t x);
typedef mp_float_t(*uumpy_unary_float2_func)(mp_float_t x, mp_float_t y);

typedef struct _uumpy_universal_spec {
    mp_int_t layers:8; // Number of dimensions unrolled in this function
    mp_int_t value_size:8;
    mp_int_t _padding:16;
    union {
        uumpy_universal_binary binary;
        uumpy_universal_unary unary;
    } apply_fn;
    union {
        mp_unary_op_t u_op;
        mp_binary_op_t b_op;
        uumpy_unary_float_func f_func;
        uumpy_unary_float2_func f2_func;
        uumpy_reduction_spec *r_spec;
        size_t c_count;
    } extra;
    void *context;
    size_t *indicies;
} uumpy_universal_spec;

// Functions for applying functions across an array
bool ufunc_apply_unary(uumpy_obj_ndarray_t *dest,
                       uumpy_obj_ndarray_t *src,
                       uumpy_universal_spec *spec);

bool ufunc_apply_binary(uumpy_obj_ndarray_t *dest,
                        uumpy_obj_ndarray_t *src1,
                        uumpy_obj_ndarray_t *src2,
                        struct _uumpy_universal_spec *spec);

// Fallback for multiply-accumulate
void ufunc_mul_acc_fallback(uumpy_obj_ndarray_t *dest, size_t dest_offset,
                            uumpy_obj_ndarray_t *src1, size_t src1_offset, size_t src1_dim,
                            uumpy_obj_ndarray_t *src2, size_t src2_offset, size_t src2_dim);

void ufunc_find_binary_op_spec(uumpy_obj_ndarray_t *src1, uumpy_obj_ndarray_t *src2,
                               char *dest_type_in_out, mp_binary_op_t op,
                               uumpy_universal_spec *spec_out);

void ufunc_find_unary_op_spec(uumpy_obj_ndarray_t *src,
                              char *dest_type_in_out, mp_unary_op_t op,
                              uumpy_universal_spec *spec_out);

void ufunc_find_copy_spec(uumpy_obj_ndarray_t *src,
                          uumpy_obj_ndarray_t *dest,
                          char *dest_type_out,
                          uumpy_universal_spec *spec_out);

void ufunc_find_unary_float_func_spec(uumpy_obj_ndarray_t *src,
                                      char *dest_type_in_out, uumpy_unary_float_func f,
                                      uumpy_universal_spec *spec_out);
