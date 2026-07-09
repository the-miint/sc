/* sc C-API integration harness (M9).
 *
 * Exercises the shipped ABI end-to-end from C: build a COO table + Utf8 targets
 * as Arrow C Data Interface structs, fit a classifier, predict, serialize,
 * deserialize, predict again, and assert the reloaded predictions match the
 * pre-serialize ones (the round-trip property). Then a forced-error call
 * (x/y sample mismatch) must return a negative status with a non-null
 * last-error string. Every array/buffer/handle is freed so the run is
 * valgrind-clean. The "== the Rust path" equality is proven in the Rust FFI
 * tests; this side is the ABI/link/leak proof.
 *
 * Build (see the Makefile `ctest` target):
 *   cc -std=c11 -I<repo>/include ctest/harness.c -L<target> -lsc_capi ... -o harness
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sc.h"

/* ---- Arrow C Data Interface producers (trivial, self-freeing) ------------ */

static void release_array(struct ArrowArray *a) {
    if (a->release == NULL) {
        return;
    }
    for (int64_t i = 0; i < a->n_buffers; i++) {
        free((void *)a->buffers[i]); /* free(NULL) for the validity slot is fine */
    }
    free((void *)a->buffers);
    a->release = NULL;
}

static void release_schema(struct ArrowSchema *s) {
    if (s->release == NULL) {
        return;
    }
    /* `format` is a string literal; nothing to free. */
    s->release = NULL;
}

static void make_primitive(const void *src, int64_t n, size_t elt, const char *format,
                           struct ArrowArray *arr, struct ArrowSchema *sch) {
    void *data = malloc((size_t)n * elt);
    memcpy(data, src, (size_t)n * elt);
    const void **buffers = malloc(2 * sizeof(void *));
    buffers[0] = NULL; /* validity: no nulls */
    buffers[1] = data;
    *arr = (struct ArrowArray){.length = n,
                               .null_count = 0,
                               .offset = 0,
                               .n_buffers = 2,
                               .n_children = 0,
                               .buffers = buffers,
                               .children = NULL,
                               .dictionary = NULL,
                               .release = release_array,
                               .private_data = NULL};
    *sch = (struct ArrowSchema){.format = format,
                                .name = NULL,
                                .metadata = NULL,
                                .flags = 0,
                                .n_children = 0,
                                .children = NULL,
                                .dictionary = NULL,
                                .release = release_schema,
                                .private_data = NULL};
}

static void make_int64_array(const int64_t *src, int64_t n, struct ArrowArray *arr,
                             struct ArrowSchema *sch) {
    make_primitive(src, n, sizeof(int64_t), "l", arr, sch);
}

static void make_float64_array(const double *src, int64_t n, struct ArrowArray *arr,
                               struct ArrowSchema *sch) {
    make_primitive(src, n, sizeof(double), "g", arr, sch);
}

static void make_utf8_array(const char *const *strs, int64_t n, struct ArrowArray *arr,
                            struct ArrowSchema *sch) {
    int32_t *offsets = malloc((size_t)(n + 1) * sizeof(int32_t));
    int32_t total = 0;
    offsets[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        total += (int32_t)strlen(strs[i]);
        offsets[i + 1] = total;
    }
    char *data = malloc(total > 0 ? (size_t)total : 1);
    int32_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        size_t l = strlen(strs[i]);
        memcpy(data + pos, strs[i], l);
        pos += (int32_t)l;
    }
    const void **buffers = malloc(3 * sizeof(void *));
    buffers[0] = NULL; /* validity */
    buffers[1] = offsets;
    buffers[2] = data;
    *arr = (struct ArrowArray){.length = n,
                               .null_count = 0,
                               .offset = 0,
                               .n_buffers = 3,
                               .n_children = 0,
                               .buffers = buffers,
                               .children = NULL,
                               .dictionary = NULL,
                               .release = release_array,
                               .private_data = NULL};
    *sch = (struct ArrowSchema){.format = "u",
                                .name = NULL,
                                .metadata = NULL,
                                .flags = 0,
                                .n_children = 0,
                                .children = NULL,
                                .dictionary = NULL,
                                .release = release_schema,
                                .private_data = NULL};
}

static void release_arr(struct ArrowArray *a) {
    if (a->release) {
        a->release(a);
    }
}
static void release_sch(struct ArrowSchema *s) {
    if (s->release) {
        s->release(s);
    }
}

/* Extract the i-th string of a consumed Utf8 output array into `out`
 * (NUL-terminated, capacity `cap`). */
static void utf8_at(const struct ArrowArray *a, int64_t i, char *out, size_t cap) {
    const int32_t *offsets = (const int32_t *)a->buffers[1];
    const char *data = (const char *)a->buffers[2];
    int32_t lo = offsets[i];
    int32_t hi = offsets[i + 1];
    size_t len = (size_t)(hi - lo);
    if (len >= cap) {
        len = cap - 1;
    }
    memcpy(out, data + lo, len);
    out[len] = '\0';
}

/* ---- Fixed dataset -------------------------------------------------------- */

/* Separable 6x2, 2-class (mirrors the Rust FFI tests). */
enum { N_SAMPLES = 6, N_FEATURES = 2, NNZ = 8 };
static const int64_t ROWS[NNZ] = {1, 2, 3, 3, 4, 4, 5, 5};
static const int64_t COLS[NNZ] = {1, 0, 0, 1, 0, 1, 0, 1};
static const double VALS[NNZ] = {1, 1, 3, 3, 3, 4, 4, 3};
static const char *const LABELS[N_SAMPLES] = {"a", "a", "a", "b", "b", "b"};

static void fill_table(sc_coo_table_t *table) {
    memset(table, 0, sizeof(*table)); /* absent sample_ids/feature_ids: release == NULL */
    make_int64_array(ROWS, NNZ, &table->rows, &table->rows_schema);
    make_int64_array(COLS, NNZ, &table->cols, &table->cols_schema);
    make_float64_array(VALS, NNZ, &table->vals, &table->vals_schema);
    table->n_samples = N_SAMPLES;
    table->n_features = N_FEATURES;
}

static void free_table(sc_coo_table_t *table) {
    release_arr(&table->rows);
    release_sch(&table->rows_schema);
    release_arr(&table->cols);
    release_sch(&table->cols_schema);
    release_arr(&table->vals);
    release_sch(&table->vals_schema);
}

static sc_rf_params_t default_params(void) {
    sc_rf_params_t p;
    memset(&p, 0, sizeof(p));
    p.criterion = SC_CRITERION_GINI;
    p.max_depth = 0; /* unbounded */
    p.max_features.kind = SC_MAX_FEATURES_SQRT;
    p.min_samples_split.kind = SC_MIN_SAMPLES_COUNT;
    p.min_samples_split.count = 2;
    p.min_samples_leaf.kind = SC_MIN_SAMPLES_COUNT;
    p.min_samples_leaf.count = 1;
    p.n_estimators = 16;
    p.bootstrap = true;
    p.max_samples.kind = SC_MAX_SAMPLES_ALL;
    p.random_state = 42;
    p.cv = 5;
    return p;
}

/* Fit → predict → return the 6 predicted labels in `out`. */
static void fit_and_predict(sc_context_t *ctx, char out[N_SAMPLES][8]) {
    sc_coo_table_t table;
    fill_table(&table);
    struct ArrowArray targets;
    struct ArrowSchema targets_schema;
    make_utf8_array(LABELS, N_SAMPLES, &targets, &targets_schema);
    sc_rf_params_t params = default_params();

    sc_model_t *model = NULL;
    sc_status_t st = sc_fit_classifier(ctx, &table, &targets, &targets_schema, &params, &model);
    assert(st == SC_OK && model != NULL);

    struct ArrowArray pred;
    struct ArrowSchema pred_s;
    st = sc_predict(ctx, model, &table, &pred, &pred_s);
    assert(st == SC_OK);
    assert(pred.length == N_SAMPLES);
    for (int64_t i = 0; i < N_SAMPLES; i++) {
        utf8_at(&pred, i, out[i], 8);
    }
    release_arr(&pred);
    release_sch(&pred_s);

    /* Serialize → deserialize → predict again, assert identical predictions. */
    uint8_t *buf = NULL;
    size_t len = 0;
    st = sc_model_serialize(model, &buf, &len);
    assert(st == SC_OK && buf != NULL && len > 0);
    sc_model_t *reloaded = NULL;
    st = sc_model_deserialize(ctx, buf, len, &reloaded);
    assert(st == SC_OK && reloaded != NULL);
    sc_buffer_free(buf, len);

    struct ArrowArray pred2;
    struct ArrowSchema pred2_s;
    st = sc_predict(ctx, reloaded, &table, &pred2, &pred2_s);
    assert(st == SC_OK);
    for (int64_t i = 0; i < N_SAMPLES; i++) {
        char after[8];
        utf8_at(&pred2, i, after, 8);
        assert(strcmp(after, out[i]) == 0 && "reloaded prediction differs");
    }
    release_arr(&pred2);
    release_sch(&pred2_s);

    sc_model_free(model);
    sc_model_free(reloaded);
    free_table(&table);
    release_arr(&targets);
    release_sch(&targets_schema);
}

/* Fit a classifier, run sc_shap, and assert the additivity identity
 * (base[o] + sum_f shap[0][o][f] == predict_proba[0][o]) for sample 0. Releases
 * every output array so the leak-check covers the new release path. */
static void exercise_shap(sc_context_t *ctx) {
    sc_coo_table_t table;
    fill_table(&table);
    struct ArrowArray targets;
    struct ArrowSchema targets_schema;
    make_utf8_array(LABELS, N_SAMPLES, &targets, &targets_schema);
    sc_rf_params_t params = default_params();

    sc_model_t *model = NULL;
    sc_status_t st = sc_fit_classifier(ctx, &table, &targets, &targets_schema, &params, &model);
    assert(st == SC_OK && model != NULL);

    /* predict_proba for the additivity comparison. */
    struct ArrowArray proba, pclasses;
    struct ArrowSchema proba_s, pclasses_s;
    st = sc_predict_proba(ctx, model, &table, &proba, &proba_s, &pclasses, &pclasses_s);
    assert(st == SC_OK);
    const int64_t n_classes = pclasses.length;
    const double *proba_vals = (const double *)proba.children[0]->buffers[1];

    /* TreeSHAP. */
    sc_shap_result_t shap;
    st = sc_shap(ctx, model, &table, &shap);
    assert(st == SC_OK);
    assert(shap.n_outputs == n_classes);
    assert(shap.n_features == N_FEATURES);
    assert(shap.shap_values.length == N_SAMPLES);
    const int64_t width = shap.n_outputs * shap.n_features;
    /* FixedSizeList: flat Float64 values live in the single child array. */
    const double *sv = (const double *)shap.shap_values.children[0]->buffers[1];
    const double *base = (const double *)shap.base_values.buffers[1];

    for (int64_t o = 0; o < shap.n_outputs; o++) {
        double row_sum = 0.0;
        for (int64_t f = 0; f < shap.n_features; f++) {
            row_sum += sv[0 * width + o * shap.n_features + f];
        }
        double recon = base[o] + row_sum;
        double want = proba_vals[0 * n_classes + o];
        assert(fabs(recon - want) < 1e-9 && "shap additivity");
    }
    printf("  shap: n_outputs=%lld n_features=%lld additive OK\n", (long long)shap.n_outputs,
           (long long)shap.n_features);

    /* Release every output array (arrow-rs callbacks recurse into children). */
    release_arr(&shap.shap_values);
    release_sch(&shap.shap_values_schema);
    release_arr(&shap.base_values);
    release_sch(&shap.base_values_schema);
    release_arr(&shap.classes);
    release_sch(&shap.classes_schema);
    release_arr(&proba);
    release_sch(&proba_s);
    release_arr(&pclasses);
    release_sch(&pclasses_s);

    sc_model_free(model);
    free_table(&table);
    release_arr(&targets);
    release_sch(&targets_schema);
}

static void forced_error(sc_context_t *ctx) {
    sc_coo_table_t table;
    fill_table(&table);
    /* 2 labels for 6 samples → x/y mismatch. */
    const char *const bad[2] = {"a", "b"};
    struct ArrowArray targets;
    struct ArrowSchema targets_schema;
    make_utf8_array(bad, 2, &targets, &targets_schema);
    sc_rf_params_t params = default_params();

    sc_model_t *model = NULL;
    sc_status_t st = sc_fit_classifier(ctx, &table, &targets, &targets_schema, &params, &model);
    assert(st < 0 && "expected a negative status");
    assert(model == NULL);
    const char *msg = sc_context_last_error(ctx);
    assert(msg != NULL && "expected a last-error message");
    printf("  forced error: status=%d msg=\"%s\"\n", st, msg);

    free_table(&table);
    release_arr(&targets);
    release_sch(&targets_schema);
}

int main(void) {
    sc_context_t *ctx = NULL;
    sc_status_t st = sc_context_new(NULL, &ctx);
    assert(st == SC_OK && ctx != NULL);

    char preds[N_SAMPLES][8];
    fit_and_predict(ctx, preds);
    printf("  predictions:");
    for (int i = 0; i < N_SAMPLES; i++) {
        printf(" %s", preds[i]);
    }
    printf("\n");
    /* Separable data: first three 'a', last three 'b'. */
    for (int i = 0; i < 3; i++) {
        assert(strcmp(preds[i], "a") == 0);
    }
    for (int i = 3; i < N_SAMPLES; i++) {
        assert(strcmp(preds[i], "b") == 0);
    }

    exercise_shap(ctx);
    forced_error(ctx);

    sc_context_free(ctx);
    printf("harness: OK\n");
    return 0;
}
