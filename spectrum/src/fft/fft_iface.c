/* fft_iface.c - FFT abstraction with LRU plan cache over kissfft.
 *
 * An FFTW build can be introduced later by swapping the engine functions
 * behind the same interface (SPECTRUM_USE_FFTW); the default engine is the
 * vendored BSD-3 kissfft. Conventions match numpy (see fft_iface.h).
 */
#include "fft_iface.h"
#include "../common.h"

#include <complex.h>
#include <pthread.h>

#include "kissfft/kiss_fft.h"

typedef struct plan_node {
    size_t n;
    int inverse;
    kiss_fft_cfg cfg;
    struct plan_node *prev, *next;
    unsigned long stamp;
} plan_node;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static plan_node *g_head = NULL, *g_tail = NULL;
static unsigned long g_clock = 0;
#define SP_FFT_CACHE_MAX 64

static size_t count_plans_locked(void) {
    size_t c = 0;
    plan_node *p;
    for (p = g_head; p; p = p->next) c++;
    return c;
}

static void evict_lru_locked(void) {
    while (count_plans_locked() > SP_FFT_CACHE_MAX && g_tail) {
        plan_node *victim = g_tail;
        if (victim->prev) victim->prev->next = NULL;
        g_tail = victim->prev;
        if (!g_tail) g_head = NULL;
        kiss_fft_free(victim->cfg);
        free(victim);
    }
}

static kiss_fft_cfg get_plan(size_t n, int inverse) {
    plan_node *found = NULL, *node;
    kiss_fft_cfg cfg;
    plan_node *p;

    pthread_mutex_lock(&g_lock);
    for (p = g_head; p; p = p->next) {
        if (p->n == n && p->inverse == inverse) {
            found = p;
            break;
        }
    }
    if (found) {
        found->stamp = ++g_clock;
        if (found != g_head) {
            if (found->prev) found->prev->next = found->next;
            if (found->next) found->next->prev = found->prev;
            if (found == g_tail) g_tail = found->prev;
            found->prev = NULL;
            found->next = g_head;
            if (g_head) g_head->prev = found;
            g_head = found;
        }
        pthread_mutex_unlock(&g_lock);
        return found->cfg;
    }
    pthread_mutex_unlock(&g_lock);

    cfg = kiss_fft_alloc((int)n, inverse, NULL, NULL);
    if (!cfg) return NULL;

    node = (plan_node *)calloc(1, sizeof(plan_node));
    if (!node) {
        kiss_fft_free(cfg);
        return NULL;
    }
    node->n = n;
    node->inverse = inverse;
    node->cfg = cfg;

    pthread_mutex_lock(&g_lock);
    /* re-check in case another thread inserted the same size meanwhile */
    for (p = g_head; p; p = p->next) {
        if (p->n == n && p->inverse == inverse) {
            found = p;
            break;
        }
    }
    if (found) {
        pthread_mutex_unlock(&g_lock);
        kiss_fft_free(cfg);
        free(node);
        return found->cfg;
    }
    node->stamp = ++g_clock;
    node->next = g_head;
    if (g_head) g_head->prev = node;
    g_head = node;
    if (!g_tail) g_tail = node;
    evict_lru_locked();
    pthread_mutex_unlock(&g_lock);
    return cfg;
}

void sp_fft_clear_cache(void) {
    plan_node *p;
    pthread_mutex_lock(&g_lock);
    p = g_head;
    while (p) {
        plan_node *nx = p->next;
        kiss_fft_free(p->cfg);
        free(p);
        p = nx;
    }
    g_head = g_tail = NULL;
    pthread_mutex_unlock(&g_lock);
}

int sp_fft_forward(double complex *data, size_t n) {
    kiss_fft_cfg cfg = get_plan(n, 0);
    kiss_fft_cpx *tmp;
    if (!cfg) return 1;
    tmp = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * n);
    if (!tmp) return 1;
    kiss_fft(cfg, (const kiss_fft_cpx *)data, tmp);
    memcpy(data, tmp, sizeof(double complex) * n);
    free(tmp);
    return 0;
}

int sp_fft_backward(double complex *data, size_t n) {
    kiss_fft_cfg cfg = get_plan(n, 1);
    kiss_fft_cpx *tmp;
    const double inv = 1.0 / (double)n;
    size_t i;
    if (!cfg) return 1;
    tmp = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * n);
    if (!tmp) return 1;
    kiss_fft(cfg, (const kiss_fft_cpx *)data, tmp);
    for (i = 0; i < n; i++) {
        data[i] = ((double complex *)tmp)[i] * inv;
    }
    free(tmp);
    return 0;
}
