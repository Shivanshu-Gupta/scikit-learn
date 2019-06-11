cimport numpy as np


cdef extern from "src/liblinear/linear.h":
    cdef struct feature_node
    cdef struct problem
    cdef struct model
    cdef struct parameter
    ctypedef problem* problem_const_ptr "problem const *"
    ctypedef parameter* parameter_const_ptr "parameter const *"
    ctypedef char* char_const_ptr "char const *"
    char_const_ptr check_parameter(problem_const_ptr prob, parameter_const_ptr param)
    model *train(problem_const_ptr prob, parameter_const_ptr param) nogil
    int get_nr_feature (model *model)
    int get_nr_class (model *model)
    void get_n_iter (model *model, int *n_iter)
    void free_and_destroy_model (model **)
    void destroy_param (parameter *)

cdef extern from "src/liblinear/liblinear_helper.c":
    void copy_w(void *, model *, int)
    parameter *set_parameter(int, double, double, int, char *, char *, int, int, double)
    problem *set_problem (char *, char *, np.npy_intp *, double, char *)
    problem *csr_set_problem (char *values, np.npy_intp *n_indices,
        char *indices, np.npy_intp *n_indptr, char *indptr, char *Y,
        np.npy_intp n_features, double bias, char *)

    problem *dense_dense_set_problem_parabel (char *X0, char *X1,
        np.npy_intp n_samples, char *Y, np.npy_intp *dims0, np.npy_intp *dims1,
        double bias, int concat, char *iX0, char *iX1, double full0, char *sample_weight)
    problem *csr_dense_set_problem_parabel (
        char *values0, np.npy_intp *n_indices0, char *indices0, np.npy_intp *n_indptr0, char *indptr0,
        char *X1,
        np.npy_intp n_samples, char *Y,
        np.npy_intp n_features0, np.npy_intp *dims1,
        double bias, int concat, char *iX0, char *iX1, double full0, char *sample_weight)
    problem *dense_csr_set_problem_parabel (
        char *X0,
        char *values1, np.npy_intp *n_indices1, char *indices1, np.npy_intp *n_indptr1, char *indptr1,
        np.npy_intp n_samples, char *Y,
        np.npy_intp *dims0, np.npy_intp n_features1,
        double bias, int concat, char *iX0, char *iX1, double full0, char *sample_weight)
    problem *csr_csr_set_problem_parabel (char *values0, np.npy_intp *n_indices0,
        char *indices0, np.npy_intp *n_indptr0, char *indptr0, char *values1,
        np.npy_intp *n_indices1, char *indices1, np.npy_intp *n_indptr1, char *indptr1,
        np.npy_intp n_samples, char *Y, np.npy_intp n_features0, np.npy_intp n_features1,
        double bias, int concat, char *iX0, char *iX1, double full0, char *sample_weight)

    model *set_model(parameter *, char *, np.npy_intp *, char *, double)

    double get_bias(model *)
    void free_problem (problem *)
    void free_parameter (parameter *)
    void set_verbosity(int)
