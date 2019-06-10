"""
Wrapper for liblinear

Author: fabian.pedregosa@inria.fr
"""

import  numpy as np
cimport numpy as np
cimport liblinear

np.import_array()


def train_wrap(X, np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
               bint is_sparse, int solver_type, double eps, double bias,
               double C, np.ndarray[np.float64_t, ndim=1] class_weight,
               int max_iter, unsigned random_seed, double epsilon,
               np.ndarray[np.float64_t, ndim=1, mode='c'] sample_weight):
    cdef parameter *param
    cdef problem *problem
    cdef model *model
    cdef char_const_ptr error_msg
    cdef int len_w

    if is_sparse:
        problem = csr_set_problem(
                (<np.ndarray[np.float64_t, ndim=1, mode='c']>X.data).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indices).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indices).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X.indptr).data,
                Y.data, (<np.int32_t>X.shape[1]), bias,
                sample_weight.data)
    else:
        problem = set_problem(
                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).data,
                Y.data,
                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X).shape,
                bias, sample_weight.data)

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.intc)
    param = set_parameter(solver_type, eps, C, class_weight.shape[0],
                          class_weight_label.data, class_weight.data,
                          max_iter, random_seed, epsilon)

    error_msg = check_parameter(problem, param)
    if error_msg:
        free_problem(problem)
        free_parameter(param)
        raise ValueError(error_msg)

    # early return
    with nogil:
        model = train(problem, param)

    # coef matrix holder created as fortran since that's what's used in liblinear
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] w
    cdef int nr_class = get_nr_class(model)

    cdef int labels_ = nr_class
    if nr_class == 2:
        labels_ = 1
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] n_iter = np.zeros(labels_, dtype=np.intc)
    get_n_iter(model, <int *>n_iter.data)

    cdef int nr_feature = get_nr_feature(model)
    if bias > 0: nr_feature = nr_feature + 1
    if nr_class == 2 and solver_type != 4:  # solver is not Crammer-Singer
        w = np.empty((1, nr_feature),order='F')
        copy_w(w.data, model, nr_feature)
    else:
        len_w = (nr_class) * nr_feature
        w = np.empty((nr_class, nr_feature),order='F')
        copy_w(w.data, model, len_w)

    ### FREE
    free_and_destroy_model(&model)
    free_problem(problem)
    free_parameter(param)
    # destroy_param(param)  don't call this or it will destroy class_weight_label and class_weight

    return w, n_iter


def train_wrap_parabel(X0, X1, np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
               bint is_sparse0, bint is_sparse1, int solver_type, double eps, double bias,
               double C, np.ndarray[np.float64_t, ndim=1] class_weight,
               int max_iter, unsigned random_seed, double epsilon, int concat,
               pairs, double full0, np.ndarray[np.float64_t, ndim=1, mode='c'] sample_weight):

    DEBUG = False
    cdef parameter *param
    cdef problem *problem
    cdef model *model
    cdef char_const_ptr error_msg
    cdef int len_w

    # cdef np.ndarray[np.uint64_t,   ndim=1, mode='c'] cpairs
    cdef char *cpairs

    if pairs is None:
        cpairs = NULL
    else:
        cpairs = (<np.ndarray[np.uint64_t,   ndim=1, mode='c']>pairs).data
    if is_sparse0 and is_sparse1:
        # print("here")
        problem = csr_csr_set_problem_parabel(
                (<np.ndarray[np.float64_t, ndim=1, mode='c']>X0.data).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indices).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indices).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indptr).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indptr).data,
                (<np.ndarray[np.float64_t, ndim=1, mode='c']>X1.data).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indices).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indices).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indptr).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indptr).data,
                (<np.int32_t>Y.shape[0]), Y.data,
                (<np.int32_t>X0.shape[1]),
                (<np.int32_t>X1.shape[1]),
                bias, concat, cpairs, full0, sample_weight.data)
    elif is_sparse0 and not is_sparse1:
        problem = csr_dense_set_problem_parabel(
                (<np.ndarray[np.float64_t, ndim=1, mode='c']>X0.data).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indices).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indices).data,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indptr).shape,
                (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X0.indptr).data,
                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X1).data,
                (<np.int32_t>Y.shape[0]), Y.data,
                (<np.int32_t>X0.shape[1]),
                (<np.ndarray[np.int32_t, ndim=2, mode='c']>X1).shape,
                bias, concat, cpairs, full0, sample_weight.data)
    elif not is_sparse0 and is_sparse1:
            problem = dense_csr_set_problem_parabel(
                    (<np.ndarray[np.float64_t, ndim=2, mode='c']>X0).data,
                    (<np.ndarray[np.float64_t, ndim=1, mode='c']>X1.data).data,
                    (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indices).shape,
                    (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indices).data,
                    (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indptr).shape,
                    (<np.ndarray[np.int32_t,   ndim=1, mode='c']>X1.indptr).data,
                    (<np.int32_t>Y.shape[0]), Y.data,
                    (<np.ndarray[np.int32_t, ndim=2, mode='c']>X0).shape,
                    (<np.int32_t>X1.shape[1]),
                    bias, concat, cpairs, full0, sample_weight.data)
    else:
        problem = dense_dense_set_problem_parabel(
                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X0).data,
                (<np.ndarray[np.float64_t, ndim=2, mode='c']>X1).data,
                (<np.int32_t>Y.shape[0]), Y.data,
                (<np.ndarray[np.int32_t, ndim=2, mode='c']>X0).shape,
                (<np.ndarray[np.int32_t, ndim=2, mode='c']>X1).shape,
                bias, concat, cpairs, full0, sample_weight.data)

    if DEBUG:
        print("problem created")

    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] \
        class_weight_label = np.arange(class_weight.shape[0], dtype=np.intc)
    param = set_parameter(solver_type, eps, C, class_weight.shape[0],
                          class_weight_label.data, class_weight.data,
                          max_iter, random_seed, epsilon)
    error_msg = check_parameter(problem, param)
    # print("here")
    if error_msg:
        free_problem(problem)
        free_parameter(param)
        raise ValueError(error_msg)
    # early return
    with nogil:
        model = train(problem, param)

    # coef matrix holder created as fortran since that's what's used in liblinear
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] w
    cdef int nr_class = get_nr_class(model)

    cdef int labels_ = nr_class
    if nr_class == 2:
        labels_ = 1
    cdef np.ndarray[np.int32_t, ndim=1, mode='c'] n_iter = np.zeros(labels_, dtype=np.intc)
    get_n_iter(model, <int *>n_iter.data)

    cdef int nr_feature = get_nr_feature(model)
    if bias > 0: nr_feature = nr_feature + 1
    if nr_class == 2 and solver_type != 4:  # solver is not Crammer-Singer
        w = np.empty((1, nr_feature),order='F')
        copy_w(w.data, model, nr_feature)
    else:
        len_w = (nr_class) * nr_feature
        w = np.empty((nr_class, nr_feature),order='F')
        copy_w(w.data, model, len_w)

    ### FREE
    free_and_destroy_model(&model)
    free_problem(problem)
    free_parameter(param)
    # destroy_param(param)  don't call this or it will destroy class_weight_label and class_weight

    return w, n_iter


def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of libsvm library
    """
    set_verbosity(verbosity)
