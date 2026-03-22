/*
 * DCFR solver core — optimized C with BLAS for matrix operations.
 * Compile (macOS): gcc -O3 -march=native -ffast-math -shared -fPIC -o dcfr_core.so dcfr_core.c -framework Accelerate
 * Compile (Linux): gcc -O3 -march=native -ffast-math -shared -fPIC -o dcfr_core.so dcfr_core.c -lopenblas -lm
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* BLAS support: compile with -DUSE_BLAS to enable.
 * Without it, uses optimized scalar loops (still fast with -O3). */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#define HAS_BLAS 1
#else
#define HAS_BLAS 0
#endif

#if !HAS_BLAS
/* Scalar fallback for dgemv */
static void my_dgemv_notrans(int M, int N, const double *A, int lda,
                              const double *x, double *y) {
    for (int i = 0; i < M; i++) {
        double sum = 0;
        const double *row = &A[i * lda];
        for (int j = 0; j < N; j++)
            sum += row[j] * x[j];
        y[i] = sum;
    }
}
static void my_dgemv_trans(int M, int N, const double *A, int lda,
                            const double *x, double *y) {
    memset(y, 0, N * sizeof(double));
    for (int i = 0; i < M; i++) {
        const double *row = &A[i * lda];
        double xi = x[i];
        for (int j = 0; j < N; j++)
            y[j] += row[j] * xi;
    }
}
#endif

#define MAX_ACT 7
#define MAX_DEPTH 15

typedef struct {
    double *strategy;     /* [max_hands * MAX_ACT] */
    double *action_vals;  /* [MAX_ACT * ho] */
    double *new_reach;    /* [max_hands] */
    double *diff;         /* [ho] scratch for regret computation */
} DepthBuffers;

static void regret_match(const double *reg, int n_hands, int nact,
                          double *strategy) {
    for (int h = 0; h < n_hands; h++) {
        double total = 0;
        for (int a = 0; a < nact; a++) {
            double v = reg[h * MAX_ACT + a];
            double pos = v > 0 ? v : 0;
            strategy[h * MAX_ACT + a] = pos;
            total += pos;
        }
        if (total > 0) {
            double inv = 1.0 / total;
            for (int a = 0; a < nact; a++)
                strategy[h * MAX_ACT + a] *= inv;
        } else {
            double u = 1.0 / nact;
            for (int a = 0; a < nact; a++)
                strategy[h * MAX_ACT + a] = u;
        }
    }
}

static void cfr_traverse(
    int node_id, int depth,
    const double * __restrict__ hero_reach,
    const double * __restrict__ opp_reach,
    const int *player,
    const int *n_actions,
    const int *children,
    const int *hero_idx,
    const int *opp_idx,
    const int *term_idx,
    const double *tv,
    double * __restrict__ hero_regrets,
    double * __restrict__ hero_strat_sum,
    double * __restrict__ opp_regrets,
    int n_hero, int n_opp,
    DepthBuffers *bufs,
    double * __restrict__ node_value
) {
    int ho = n_hero * n_opp;

    if (term_idx[node_id] >= 0) {
        memcpy(node_value, &tv[term_idx[node_id] * ho], ho * sizeof(double));
        return;
    }

    int nact = n_actions[node_id];
    int p = player[node_id];
    double *strategy = bufs[depth].strategy;
    double *action_vals = bufs[depth].action_vals;
    double *new_reach = bufs[depth].new_reach;

    if (p == 0) {
        int idx = hero_idx[node_id];
        regret_match(&hero_regrets[idx * n_hero * MAX_ACT], n_hero, nact, strategy);
        memset(node_value, 0, ho * sizeof(double));

        for (int a = 0; a < nact; a++) {
            int child = children[node_id * MAX_ACT + a];
            double *av = &action_vals[a * ho];

            for (int h = 0; h < n_hero; h++)
                new_reach[h] = hero_reach[h] * strategy[h * MAX_ACT + a];

            cfr_traverse(child, depth + 1, new_reach, opp_reach,
                         player, n_actions, children,
                         hero_idx, opp_idx, term_idx,
                         tv, hero_regrets, hero_strat_sum, opp_regrets,
                         n_hero, n_opp, bufs, av);

            for (int h = 0; h < n_hero; h++) {
                double s = strategy[h * MAX_ACT + a];
                int base = h * n_opp;
                for (int o = 0; o < n_opp; o++)
                    node_value[base + o] += s * av[base + o];
            }
        }

        /* Regret update using BLAS:
         * For each action a:
         *   diff[h,o] = av[a,h,o] - nv[h,o]    (elementwise)
         *   cf_regret[h] = diff[h,:] @ opp_reach  (matrix-vector: dgemv)
         */
        double *reg = &hero_regrets[idx * n_hero * MAX_ACT];
        double *diff = bufs[depth].diff;
        for (int a = 0; a < nact; a++) {
            double *av = &action_vals[a * ho];
            for (int i = 0; i < ho; i++)
                diff[i] = av[i] - node_value[i];

            /* diff is (n_hero × n_opp), opp_reach is (n_opp,)
             * result = diff @ opp_reach → (n_hero,)
             * cblas_dgemv: y = alpha * A * x + beta * y
             * A = diff (n_hero rows × n_opp cols, row-major) */
#if HAS_BLAS
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        n_hero, n_opp, 1.0,
                        diff, n_opp,
                        opp_reach, 1,
                        0.0, new_reach, 1);
#else
            my_dgemv_notrans(n_hero, n_opp, diff, n_opp,
                             opp_reach, new_reach);
#endif

            for (int h = 0; h < n_hero; h++)
                reg[h * MAX_ACT + a] += new_reach[h];
        }

        /* Strategy sum update */
        double *ss = &hero_strat_sum[idx * n_hero * MAX_ACT];
        for (int h = 0; h < n_hero; h++)
            for (int a = 0; a < nact; a++)
                ss[h * MAX_ACT + a] += hero_reach[h] * strategy[h * MAX_ACT + a];

    } else {
        int idx = opp_idx[node_id];
        regret_match(&opp_regrets[idx * n_opp * MAX_ACT], n_opp, nact, strategy);
        memset(node_value, 0, ho * sizeof(double));

        for (int a = 0; a < nact; a++) {
            int child = children[node_id * MAX_ACT + a];
            double *av = &action_vals[a * ho];

            for (int o = 0; o < n_opp; o++)
                new_reach[o] = opp_reach[o] * strategy[o * MAX_ACT + a];

            cfr_traverse(child, depth + 1, hero_reach, new_reach,
                         player, n_actions, children,
                         hero_idx, opp_idx, term_idx,
                         tv, hero_regrets, hero_strat_sum, opp_regrets,
                         n_hero, n_opp, bufs, av);

            for (int h = 0; h < n_hero; h++) {
                int base = h * n_opp;
                for (int o = 0; o < n_opp; o++)
                    node_value[base + o] +=
                        strategy[o * MAX_ACT + a] * av[base + o];
            }
        }

        /* Opponent regret update with BLAS:
         * cf_regret[o, a] = sum_h (nv[h,o] - av[a,h,o]) * hero_reach[h]
         * diff[h,o] = nv[h,o] - av[a,h,o]
         * cf_regret[:, a] = diff^T @ hero_reach  (transpose matrix-vector)
         */
        double *reg = &opp_regrets[idx * n_opp * MAX_ACT];
        double *diff = bufs[depth].diff;
        for (int a = 0; a < nact; a++) {
            double *av = &action_vals[a * ho];
            for (int i = 0; i < ho; i++)
                diff[i] = node_value[i] - av[i];

            /* diff is (n_hero × n_opp), hero_reach is (n_hero,)
             * result = diff^T @ hero_reach → (n_opp,)
             * cblas_dgemv with CblasTrans */
#if HAS_BLAS
            cblas_dgemv(CblasRowMajor, CblasTrans,
                        n_hero, n_opp, 1.0,
                        diff, n_opp,
                        hero_reach, 1,
                        0.0, new_reach, 1);
#else
            my_dgemv_trans(n_hero, n_opp, diff, n_opp,
                           hero_reach, new_reach);
#endif

            for (int o = 0; o < n_opp; o++)
                reg[o * MAX_ACT + a] += new_reach[o];
        }
    }
}

void run_dcfr_c(
    const int *player, const int *n_actions, const int *children,
    const int *hero_idx, const int *opp_idx, const int *term_idx,
    int n_nodes,
    const double *tv,
    int n_hero, int n_opp,
    int n_hero_nodes, int n_opp_nodes,
    const double *opp_weights,
    int iterations,
    double *hero_strat_sum,
    double *root_game_value  /* [n_hero * n_opp] output: game value at root */
) {
    int hr_size = n_hero_nodes * n_hero * MAX_ACT;
    int or_size = n_opp_nodes * n_opp * MAX_ACT;
    int ho = n_hero * n_opp;
    int max_hands = (n_hero > n_opp) ? n_hero : n_opp;

    double *hero_regrets = (double *)calloc(hr_size, sizeof(double));
    double *opp_regrets = (double *)calloc(or_size, sizeof(double));
    memset(hero_strat_sum, 0, hr_size * sizeof(double));

    DepthBuffers bufs[MAX_DEPTH];
    for (int d = 0; d < MAX_DEPTH; d++) {
        bufs[d].strategy = (double *)malloc(max_hands * MAX_ACT * sizeof(double));
        bufs[d].action_vals = (double *)malloc(MAX_ACT * ho * sizeof(double));
        bufs[d].new_reach = (double *)malloc(max_hands * sizeof(double));
        bufs[d].diff = (double *)malloc(ho * sizeof(double));
    }

    double *hero_reach = (double *)malloc(n_hero * sizeof(double));
    double *opp_reach = (double *)malloc(n_opp * sizeof(double));
    double *node_value = (double *)malloc(ho * sizeof(double));

    for (int t = 1; t <= iterations; t++) {
        if (t > 1) {
            double tm1 = (double)(t - 1);
            double pos_w = pow(tm1, 1.5) / (pow(tm1, 1.5) + 1.0);
            double strat_w = pow(tm1 / (double)t, 2.0);

            for (int i = 0; i < hr_size; i++)
                hero_regrets[i] *= (hero_regrets[i] > 0) ? pos_w : 0.0;
            for (int i = 0; i < or_size; i++)
                opp_regrets[i] *= (opp_regrets[i] > 0) ? pos_w : 0.0;
            for (int i = 0; i < hr_size; i++)
                hero_strat_sum[i] *= strat_w;
        }

        for (int h = 0; h < n_hero; h++)
            hero_reach[h] = 1.0 / n_hero;
        for (int o = 0; o < n_opp; o++)
            opp_reach[o] = opp_weights[o];

        cfr_traverse(0, 0, hero_reach, opp_reach,
                     player, n_actions, children,
                     hero_idx, opp_idx, term_idx,
                     tv, hero_regrets, hero_strat_sum, opp_regrets,
                     n_hero, n_opp, bufs, node_value);
    }

    /* Copy root game value to output */
    if (root_game_value != NULL) {
        memcpy(root_game_value, node_value, ho * sizeof(double));
    }

    for (int d = 0; d < MAX_DEPTH; d++) {
        free(bufs[d].strategy);
        free(bufs[d].action_vals);
        free(bufs[d].new_reach);
        free(bufs[d].diff);
    }
    free(hero_regrets);
    free(opp_regrets);
    free(hero_reach);
    free(opp_reach);
    free(node_value);
}
