/*
 * This file is part of the uumpy project
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Nicko van Someren
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

#include "py/runtime.h"
#include "py/binary.h"

#include "moduumpy.h"
#include "ufunc.h"

#define UUMPY_REDUCTION_FLAG_BOOL_OUT (0x100) // Result is boolean no matter what the input is
#define UUMPY_REDUCTION_FLAG_INT_OUT (0x200) // Result is integer no matter what the input is
#define UUMPY_REDUCTION_FLAG_FLOAT_COMPLEX_OUT (0x400) // Result is float unless the input is complex, in which case it's complex
#define UUMPY_REDUCTION_FLAG_OUT_TYPE_FIXED (0xf00) // The result type is (largely) invariant of the input type

#define UUMPY_REDUCTION_FLAG_1D_ONLY (0x1000) // Can only be applied to a single axis

#define UUMPY_REDUCTION_OP_MAX (0)
#define UUMPY_REDUCTION_OP_MIN (1)
#define UUMPY_REDUCTION_OP_SUM (2)
#define UUMPY_REDUCTION_OP_PROD (3)
#define UUMPY_REDUCTION_OP_AVERAGE (4)
#define UUMPY_REDUCTION_OP_STD (5)
#define UUMPY_REDUCTION_OP_ANY (6 | UUMPY_REDUCTION_FLAG_BOOL_OUT)
#define UUMPY_REDUCTION_OP_ALL (7 | UUMPY_REDUCTION_FLAG_BOOL_OUT)
#define UUMPY_REDUCTION_OP_ARGMAX (8 | UUMPY_REDUCTION_FLAG_1D_ONLY | UUMPY_REDUCTION_FLAG_INT_OUT)
#define UUMPY_REDUCTION_OP_ARGMIN (9 | UUMPY_REDUCTION_FLAG_1D_ONLY | UUMPY_REDUCTION_FLAG_INT_OUT)

#define UUMP_REDUCTION_FASTPATH_NONE (0)
#define UUMP_REDUCTION_FASTPATH_FLOAT (1)


#define UUMPY_REDUCTION_FUN(name, operation) \
    static mp_obj_t uumpy_reduction_ ## name(size_t n_args, const mp_obj_t *args, mp_map_t *kwargs) { \
        return uumpy_reduction_helper_1(operation, n_args, args, kwargs); \
    } \
    MP_DEFINE_CONST_FUN_OBJ_KW(uumpy_reduction_ ## name ## _obj, 1, uumpy_reduction_ ## name)


// When this is called dest points to elements where the reductions are placed
// and the reduction is applied across all remaining layers in the source
static bool ufunc_apply_reduction_unary(size_t depth,
                                        uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                        uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                        struct _uumpy_universal_spec *spec) {
    mp_int_t reduce_dim_count = src->dim_count - depth;
    mp_int_t *src_indices = spec->indices + depth;
    bool first = true;
    uumpy_reduction_spec *r_spec = spec->extra.r_spec;
    uumpy_reduction_unary r_func = r_spec->iter_func;

    void *state = spec->context;
    int count = 0;

    if (r_spec->init_func) {
        r_spec->init_func(dest, dest_offset, src, src_offset, spec, state);
    }
    
    if (reduce_dim_count == 1) {
        mp_int_t stride = src->dim_info[depth].stride;
        mp_int_t ll = src->dim_info[depth].length;
        
        for (mp_int_t i=0; i < ll; i++) {
            src_indices[0] = i;
            r_func(dest, dest_offset, src, src_offset, spec, state, first);
            src_offset += stride;
            first = false;
        }
        count = ll;
    } else {
        mp_int_t l;
        uumpy_dim_info * dim_info = src->dim_info + depth;

        for (mp_int_t i=0; i < reduce_dim_count; i++) {
            src_indices[i] = 0;
        }
        
        do {
            // DEBUG_printf("applying reduction with src_offset=%d\n", src_offset);
            r_func(dest, dest_offset, src, src_offset, spec, state, first);
            first = false;
            count += 1;
            
            for (l = reduce_dim_count-1; l >=0; l--) {
                src_offset += dim_info[l].stride;
                src_indices[l]++;

                if (src_indices[l] < dim_info[l].length) {
                    break;
                } else {
                    // Reset this row and allow moving on to the next one
                    mp_int_t ll = dim_info[l].length;
                    src_indices[l] = 0;
                    src_offset -= ll * dim_info[l].stride;
                }
            }
        } while (l >= 0);
    }
    
    r_spec->finish_func(dest, dest_offset, spec, state, count);

    return true;
}

// This function handles most dimensional reduction ops.
// The destination can be None, in which case it needs to be created.
// The axis can be an int (axis index), a tuple (taken in order), or
// None (flatten over all axes).

static mp_obj_t uumpy_reduction_generic(uumpy_obj_ndarray_t *src, uumpy_obj_ndarray_t *dest,
                                        mp_obj_t axis, uumpy_reduction_spec *spec) {
    // Determine which (and how many) axis to reduce
    // If necessary, create a view to place the reduction axis at the end
    // Set up the ufunc spec
    // If necessary, create the destination
    // Ensure that we have enough memory for the state
    // Apply the ufunc
    // Return the result

    mp_int_t reduce_layers;
    
    if (axis == mp_const_none) {
        // Reduce over all layers
        reduce_layers = src->dim_count;
    } else if (mp_obj_is_small_int(axis) ||
               mp_obj_is_type(axis, &mp_type_tuple)) {
        mp_int_t count;
        mp_obj_t *values;
        mp_int_t dim_count = src->dim_count;
        unsigned int axis_mask = 0;
        bool only_trailing = true;
        
        if (!uumpy_util_get_list_tuple(axis, &count, &values)) {
            count = 1;
            values = &axis;
        }

        if (count == 0) {
            mp_raise_ValueError(MP_ERROR_TEXT("axis tuple is empty"));
        }

        
        reduce_layers = count;

        for (int i=0; i < count; i++) {
            if (!mp_obj_is_small_int(values[i])) {
                goto axis_type_error;
            }
            
            mp_int_t a_index = MP_OBJ_SMALL_INT_VALUE(values[i]);
            if (a_index < 0) {
                a_index += dim_count;
            }

            if (a_index < 0 || a_index >= dim_count) {
                mp_raise_ValueError(MP_ERROR_TEXT("axis index out of range"));
            }

            if (axis_mask & (1 << a_index)) {
                mp_raise_ValueError(MP_ERROR_TEXT("axis can only occur once"));
            }

            axis_mask |= (1 << a_index);
            
            if (a_index != ((dim_count - count) + i)) {
                only_trailing = false;
            }
        }

        if (!only_trailing) {
            // Need to create a new view with the reduction dimensions at the end
            uumpy_dim_info new_dim_info[UUMPY_MAX_DIMS];

            mp_int_t used = 0;
            for (mp_int_t i=0; i < src->dim_count; i++) {
                if ((axis_mask & (1 << i)) == 0) {
                    new_dim_info[used] = src->dim_info[i];
                    used++;
                }
            }
            for (mp_int_t i=0; i < count; i++) {
                int j =  MP_OBJ_SMALL_INT_VALUE(values[i]);
                new_dim_info[used] = src->dim_info[j];
                used++;
            }

            src = ndarray_new_view(src, src->base_offset, src->dim_count, new_dim_info);
        }
    } else {
        goto axis_type_error;
    }
    
    uumpy_universal_spec ufn_spec = {
        .apply_fn.unary = &ufunc_apply_reduction_unary,
        .extra.r_spec = spec,
    };

    // Most of the time we don't need much context so we grab some stack
    byte context[16];
    if (spec->state_size > sizeof(context)) {
        ufn_spec.context = m_new(byte, spec->state_size);
    } else {
        ufn_spec.context = context;
    }

    if (dest) {
        // Check that the output given is the right shape

        if ((dest->dim_count != src->dim_count - reduce_layers) ||
            !ndarray_compare_dimensions_counted(src, dest, dest->dim_count)) {
            mp_raise_ValueError(MP_ERROR_TEXT("destination dimensions incompatible with result"));
        }
    } else {
        // Create the destination of the right type and shape
        dest = ndarray_new_shaped_like(spec->result_typecode, src, reduce_layers);
    }

    ufunc_apply_unary(dest, src, &ufn_spec);

    if (dest->dim_count == 0) {
        return mp_binary_get_val_array(dest->typecode, dest->data, dest->base_offset);
    } else {
        return MP_OBJ_FROM_PTR(dest);
    }
    
    axis_type_error:
        mp_raise_TypeError(MP_ERROR_TEXT("axis must be an int or tuple of ints"));
}

static void uumpy_reduction_init_zero_obj(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                          uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                          struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_obj_t *state_obj_ptr = (mp_obj_t *) state_ptr;

    *state_obj_ptr = MP_OBJ_NEW_SMALL_INT(0);
}

static void uumpy_reduction_init_one_obj(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_obj_t *state_obj_ptr = (mp_obj_t *) state_ptr;

    *state_obj_ptr = MP_OBJ_NEW_SMALL_INT(1);
}

static void uumpy_reduction_init_zero_float(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                            uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                            struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;

    *state_float_ptr = 0.0;
}

static void uumpy_reduction_init_one_float(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                           uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                           struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;

    *state_float_ptr = 1.0;
}

static void uumpy_reduction_init_zero_int(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                          uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                          struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_int_t *state_int_ptr = (mp_int_t *) state_ptr;

    *state_int_ptr = 0;
}

static void uumpy_reduction_init_one_int(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec, void *state_ptr) {
    (void) dest;
    (void) dest_offset;
    (void) src;
    (void) src_offset;
    (void) spec;

    mp_int_t *state_int_ptr = (mp_int_t *) state_ptr;

    *state_int_ptr = 1;
}

static void uumpy_reduction_finish_store_obj(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                             struct _uumpy_universal_spec *spec, void *state_ptr, int count) {
    
    (void) spec;
    
    mp_obj_t *state_obj_ptr = (mp_obj_t *) state_ptr;
    mp_binary_set_val_array(dest->typecode, dest->data, dest_offset, *state_obj_ptr);
}

static void uumpy_reduction_finish_store_float(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                               struct _uumpy_universal_spec *spec, void *state_ptr, int count) {
    
    (void) spec;
    
    ((mp_float_t *) dest->data)[dest_offset] = *((mp_float_t *) state_ptr); 
}

static void uumpy_reduction_finish_store_int(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                             struct _uumpy_universal_spec *spec, void *state_ptr, int count) {
    
    (void) spec;
    
    ((mp_int_t *) dest->data)[dest_offset] = *((mp_int_t *) state_ptr); 
}

static void uumpy_reduction_finish_store_bool(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                              struct _uumpy_universal_spec *spec, void *state_ptr, int count) {
    
    (void) spec;
    
    ((byte *) dest->data)[dest_offset] = *((byte *) state_ptr); 
}

static void uumpy_reduction_max_float_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec,
                                         void *state_ptr, bool is_first) {
    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;
    mp_float_t *src_data = (mp_float_t *) src->data;

    if ((src_data[src_offset] > *state_float_ptr) || is_first) {
        *state_float_ptr = src_data[src_offset];
    }
}

static void uumpy_reduction_min_float_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec,
                                         void *state_ptr, bool is_first) {
    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;
    mp_float_t *src_data = (mp_float_t *) src->data;

    if ((src_data[src_offset] < *state_float_ptr) || is_first) {
        *state_float_ptr = src_data[src_offset];
    }
}

static void uumpy_reduction_max_obj_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec,
                                         void *state_ptr, bool is_first) {
}

static void uumpy_reduction_min_obj_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec,
                                         void *state_ptr, bool is_first) {
}

static void uumpy_reduction_sum_float_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                         uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                         struct _uumpy_universal_spec *spec,
                                         void *state_ptr, bool is_first) {
    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;
    mp_float_t *src_data = (mp_float_t *) src->data;

    *state_float_ptr += src_data[src_offset];
}

static void uumpy_reduction_finish_average_float(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                               struct _uumpy_universal_spec *spec, void *state_ptr, int count) {
    
    ((mp_float_t *) dest->data)[dest_offset] = *((mp_float_t *) state_ptr) / count; 
}



static void uumpy_reduction_prod_float_op(uumpy_obj_ndarray_t *dest, mp_int_t dest_offset,
                                          uumpy_obj_ndarray_t *src, mp_int_t src_offset,
                                          struct _uumpy_universal_spec *spec,
                                          void *state_ptr, bool is_first) {
    mp_float_t *state_float_ptr = (mp_float_t *) state_ptr;
    mp_float_t *src_data = (mp_float_t *) src->data;

    *state_float_ptr *= src_data[src_offset];
}

static bool uumpy_reduction_find_unary_spec(unsigned int op_code,
                                            uumpy_obj_ndarray_t *src,
                                            uumpy_obj_ndarray_t *dest,
                                            uumpy_reduction_spec *spec_out) {
    mp_int_t result_typecode;

    if (op_code & UUMPY_REDUCTION_FLAG_BOOL_OUT) {
        result_typecode = 'B';
    } else if (op_code & UUMPY_REDUCTION_FLAG_INT_OUT) {
        result_typecode = 'i';
    } else if (op_code & UUMPY_REDUCTION_FLAG_FLOAT_COMPLEX_OUT) {
        // NOTE: This will need to be updated once we support complex numbers
        result_typecode = UUMPY_DEFAULT_TYPE;
    } else {
        result_typecode = src->typecode;
    }

    int fastpath_type;
    bool custom_final = false;
    
    // Check for viability of fast operations
    if ((src->typecode == UUMPY_DEFAULT_TYPE) &&
        (dest == NULL || dest->typecode == result_typecode)) {
        fastpath_type = UUMP_REDUCTION_FASTPATH_FLOAT;
    } else {
        fastpath_type = UUMP_REDUCTION_FASTPATH_NONE;
    }

    spec_out->result_typecode = result_typecode;
    // Default the context size to the size of one item
    spec_out->state_size = mp_binary_get_size('@', result_typecode, NULL);
    
    if (fastpath_type == UUMP_REDUCTION_FASTPATH_FLOAT) {
        switch (op_code) {
        case UUMPY_REDUCTION_OP_MAX:
            spec_out->init_func = NULL;
            spec_out->iter_func = &uumpy_reduction_max_float_op;
            break;
        case UUMPY_REDUCTION_OP_MIN:
            spec_out->init_func = NULL;
            spec_out->iter_func = &uumpy_reduction_min_float_op;
            break;
            
        case UUMPY_REDUCTION_OP_SUM:
            spec_out->init_func = &uumpy_reduction_init_zero_float;
            spec_out->iter_func = &uumpy_reduction_sum_float_op;
            break;
        case UUMPY_REDUCTION_OP_PROD:
            spec_out->init_func = &uumpy_reduction_init_one_float;
            spec_out->iter_func = &uumpy_reduction_prod_float_op;
            break;
        case UUMPY_REDUCTION_OP_AVERAGE:
            spec_out->init_func = &uumpy_reduction_init_zero_float;
            spec_out->iter_func = &uumpy_reduction_sum_float_op;
            spec_out->finish_func = &uumpy_reduction_finish_average_float;
            custom_final = true;
            break;
            
        case UUMPY_REDUCTION_OP_ANY:
            spec_out->init_func = &uumpy_reduction_init_zero_int;
        case UUMPY_REDUCTION_OP_ALL:
            spec_out->init_func = &uumpy_reduction_init_one_int;
        case UUMPY_REDUCTION_OP_ARGMAX:
        case UUMPY_REDUCTION_OP_ARGMIN:
        case UUMPY_REDUCTION_OP_STD:
        default:
            fastpath_type = UUMP_REDUCTION_FASTPATH_NONE;
            break;
        }
    }
    
    if (fastpath_type == UUMP_REDUCTION_FASTPATH_NONE) {
        switch (op_code) {
        case UUMPY_REDUCTION_OP_MAX:
            spec_out->init_func = NULL;
            spec_out->iter_func = &uumpy_reduction_max_obj_op;
            break;
        case UUMPY_REDUCTION_OP_MIN:
            spec_out->init_func = NULL;
            spec_out->iter_func = &uumpy_reduction_min_obj_op;
            break;

        case UUMPY_REDUCTION_OP_SUM:
            spec_out->init_func = &uumpy_reduction_init_zero_obj;
        case UUMPY_REDUCTION_OP_PROD:
            spec_out->init_func = &uumpy_reduction_init_one_obj;
        case UUMPY_REDUCTION_OP_AVERAGE:
            spec_out->init_func = &uumpy_reduction_init_zero_obj;
        case UUMPY_REDUCTION_OP_ANY:
            spec_out->init_func = &uumpy_reduction_init_zero_int;
        case UUMPY_REDUCTION_OP_ALL:
            spec_out->init_func = &uumpy_reduction_init_one_int;
        case UUMPY_REDUCTION_OP_ARGMAX:
        case UUMPY_REDUCTION_OP_ARGMIN:
        case UUMPY_REDUCTION_OP_STD:
        default:
            return false;
        }
    }

    if (!custom_final) {
        if (op_code & UUMPY_REDUCTION_FLAG_BOOL_OUT) {
            spec_out->finish_func = &uumpy_reduction_finish_store_bool;
        } else if (op_code & UUMPY_REDUCTION_FLAG_INT_OUT) {
            spec_out->finish_func = &uumpy_reduction_finish_store_int;
        } else {
            if (fastpath_type == UUMP_REDUCTION_FASTPATH_FLOAT) {
                spec_out->finish_func = &uumpy_reduction_finish_store_float;
            } else {
                spec_out->finish_func = &uumpy_reduction_finish_store_obj;
            }
        }
    }
    
    return true;
}

static mp_obj_t uumpy_reduction_helper_1(int op_code, mp_uint_t n_args,
                                         const mp_obj_t *pos_args, mp_map_t *kw_args) {
    // Parse the arguments
    enum {
        ARG_a,
        ARG_axis,
        ARG_out,
        ARG_keepdims,
    };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_a,        MP_ARG_REQUIRED | MP_ARG_OBJ,  {.u_obj  = mp_const_none} },
        { MP_QSTR_axis,     MP_ARG_OBJ,                    {.u_obj  = mp_const_none} },
        { MP_QSTR_out,      MP_ARG_KW_ONLY  | MP_ARG_OBJ,  {.u_obj  = mp_const_none} },
        { MP_QSTR_keepdims, MP_ARG_KW_ONLY  | MP_ARG_BOOL, {.u_bool = false        } },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)] = {{.u_int=0}};
    mp_arg_parse_all(n_args, pos_args, kw_args, MP_ARRAY_SIZE(allowed_args), allowed_args, args);

    uumpy_obj_ndarray_t *a = MP_OBJ_TO_PTR(args[ARG_a].u_obj);
    uumpy_obj_ndarray_t *out = (args[ARG_out].u_obj != mp_const_none) ? MP_OBJ_TO_PTR(args[ARG_out].u_obj) : NULL;
    
    if ((op_code & UUMPY_REDUCTION_FLAG_1D_ONLY) &&
        args[ARG_axis].u_obj &&
        mp_obj_is_type(args[ARG_axis].u_obj, &mp_type_tuple)) {
        mp_raise_ValueError(MP_ERROR_TEXT("axis may not be a tuple"));
    }
    
    // Find the spec
    uumpy_reduction_spec spec;
    if (!uumpy_reduction_find_unary_spec(op_code, a, out, &spec)) {
        mp_raise_NotImplementedError(MP_ERROR_TEXT("no reduction function for data type"));
    }

    // If out is given and keepdims is set, create a view for the output
    if (args[ARG_keepdims].u_bool) {
        //
        mp_raise_NotImplementedError(MP_ERROR_TEXT("keepdims not currently supported"));
    }
    
    // Execute the function
    mp_obj_t result = uumpy_reduction_generic(a, out, args[ARG_axis].u_obj, &spec);
    // TODO If keepdims is set and the result is new, reshape to restore missing dimensions

    return result;
}

UUMPY_REDUCTION_FUN(max, UUMPY_REDUCTION_OP_MAX);
UUMPY_REDUCTION_FUN(min, UUMPY_REDUCTION_OP_MIN);
UUMPY_REDUCTION_FUN(sum, UUMPY_REDUCTION_OP_SUM);
UUMPY_REDUCTION_FUN(prod, UUMPY_REDUCTION_OP_PROD);
UUMPY_REDUCTION_FUN(average, UUMPY_REDUCTION_OP_AVERAGE);
UUMPY_REDUCTION_FUN(any, UUMPY_REDUCTION_OP_ANY);
UUMPY_REDUCTION_FUN(all, UUMPY_REDUCTION_OP_ALL);
