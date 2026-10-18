/* common.h - shared utilities for the spectrum C core.
 *
 * Pure C11, no Python types. Every estimator kernel operates on plain
 * contiguous buffers and returns sp_status codes.
 */
#ifndef SPECTRUM_COMMON_H
#define SPECTRUM_COMMON_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SP_OK = 0,
    SP_ERR_ARG = 1,   /* invalid argument (shape, range, null pointer) */
    SP_ERR_MEM = 2,   /* allocation failure */
    SP_ERR_SHAPE = 3, /* inconsistent input shapes */
    SP_ERR_INTERNAL = 4
} sp_status;

/* Bump allocator arena: one free() per estimator call. */
typedef struct {
    char *base;
    size_t used;
    size_t cap;
} sp_arena;

static inline void sp_arena_init(sp_arena *a) {
    a->base = NULL;
    a->used = 0;
    a->cap = 0;
}

static inline void *sp_arena_alloc(sp_arena *a, size_t nbytes) {
    void *p;
    if (a->used + nbytes > a->cap) {
        return NULL; /* pre-sized by caller; grow is not supported */
    }
    p = a->base + a->used;
    a->used += nbytes;
    return p;
}

static inline void sp_arena_free(sp_arena *a) {
    free(a->base);
    a->base = NULL;
    a->used = 0;
    a->cap = 0;
}

/* Allocate arena with capacity; returns SP_ERR_MEM on failure. */
static inline int sp_arena_reserve(sp_arena *a, size_t cap) {
    a->base = (char *)malloc(cap ? cap : 1);
    if (!a->base) {
        a->cap = 0;
        return SP_ERR_MEM;
    }
    a->cap = cap;
    a->used = 0;
    return SP_OK;
}

/* OpenMP helpers: no-op when disabled at build time. */
#if !defined(SPECTRUM_NO_OPENMP) && defined(_OPENMP)
#include <omp.h>
#define SP_OMP_PARALLEL_FOR _Pragma("omp parallel for schedule(static)")
#define SP_OMP_MAX_THREADS() (omp_get_max_threads())
#define SP_OMP_THREAD_NUM() (omp_get_thread_num())
#else
#define SP_OMP_PARALLEL_FOR
#define SP_OMP_MAX_THREADS() (1)
#define SP_OMP_THREAD_NUM() (0)
#endif

static inline long sp_fix(double x) { /* MATLAB fix(): truncate toward zero */
    return (long)x;
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRUM_COMMON_H */
