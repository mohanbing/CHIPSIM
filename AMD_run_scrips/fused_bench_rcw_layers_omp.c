// fused_bench_rcw_layers.c  (OpenMP + per-kernel pinning)
// N-layer microbenchmark: for k = 0..L-1 do { READ_k -> COMPUTE_k -> WRITE_k }.
// Per-layer params via CSV lists: -Sr -rr -ci -Sw -rw (all length L).
// New: OpenMP thread counts per kernel (-tR, -tC, -tW) and per-kernel pin lists (--pinR/--pinC/--pinW).
// LIKWID markers per layer are preserved: "read_k", "compute_k", "write_k".
//
// Build (LIKWID + OpenMP):
//   gcc -O3 -march=native -Wall -Wextra -std=c11 fused_bench_rcw_layers.c  -o fused_bench_rcw_layers -DLIKWID_PERFMON -llikwid -fopenmp
//
// Build (no LIKWID, still supports OpenMP):
//   gcc -O3 -march=native -Wall -Wextra -std=c11 fused_bench_rcw_layers.c  -o fused_bench_rcw_layers -fopenmp
//
// Example run (2 layers, different read/compute/write sizes and iterations):
//   OMP_PROC_BIND=TRUE LIKWID_FORCE=1 likwid-perfctr -C 0-7 -g MEM  ./fused_bench_rcw_layers -L 2  -Sr 512,256 -rr 2,3 -ci 20000000,40000000 -Sw 512,256 -rw 2,3  -tR 6 -tC 4 -tW 8 --pinR 0,1,2,3,4,5 --pinC 4,5,6,7 --pinW 0,2,4,6,8,10,12,14
//
// You can alternatively use external pinning, e.g.:
//   likwid-pin -c N:0,2,4,6 -q ./fused_bench_rcw_layers ... (omit --pin* flags)
//

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_REGISTER(x)
#define LIKWID_MARKER_START(x)
#define LIKWID_MARKER_STOP(x)
#define LIKWID_MARKER_CLOSE
#endif

static volatile uint64_t g_sink = 0;

// ---- time ----
static inline uint64_t nsec_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---- pin current (calling) thread to logical core ----
static int pin_to_core(int core) {
    if (core < 0) return 0;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((unsigned)core, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        return -1;
    }
    return 0;
}

// ---- aligned alloc with prefault ----
static void* alloc_aligned(size_t bytes) {
    void* p = NULL;
    int rc = posix_memalign(&p, 4096, bytes);
    if (rc != 0) {
        errno = rc;
        perror("posix_memalign");
        return NULL;
    }
    volatile uint8_t* q = (volatile uint8_t*)p;
    for (size_t i = 0; i < bytes; i += 4096) q[i] = (uint8_t)i;
    return p;
}

// ---------- CSV parsing ----------
static int parse_csv_ull_exact(const char* s, size_t L, unsigned long long* out) {
    size_t idx = 0;
    const char* p = s ? s : "";
    if (!*p) return (L==0) ? 0 : -4;
    while (*p) {
        while (*p==' '||*p=='\t') ++p;
        errno = 0;
        char* endp = NULL;
        unsigned long long v = strtoull(p, &endp, 10);
        if (endp == p || errno != 0) return -1;
        if (idx >= L) return -2;
        out[idx++] = v;
        while (*endp==' '||*endp=='\t') ++endp;
        if (*endp == ',') { p = endp+1; continue; }
        if (*endp == '\0') { p = endp; break; }
        return -3;
    }
    return (idx == L) ? 0 : -4;
}

static int* parse_csv_int_dynamic(const char* s, size_t* out_len) {
    *out_len = 0;
    if (!s || !*s) return NULL;
    // First pass: count commas
    size_t cnt = 1;
    for (const char* p=s; *p; ++p) if (*p==',') ++cnt;
    int* arr = (int*)malloc(cnt * sizeof(int));
    if (!arr) return NULL;
    size_t idx=0;
    const char* p = s;
    while (*p) {
        while (*p==' '||*p=='\t') ++p;
        errno=0; char* endp=NULL;
        long v = strtol(p, &endp, 10);
        if (endp==p || errno!=0) { free(arr); return NULL; }
        arr[idx++] = (int)v;
        while (*endp==' '||*endp=='\t') ++endp;
        if (*endp==',') { p = endp+1; continue; }
        if (*endp=='\0') { p = endp; break; }
        free(arr); return NULL;
    }
    *out_len = idx;
    return arr;
}

// ---------- READ helpers ----------
static uint64_t read_sum_range(const uint64_t* __restrict p, size_t lo, size_t hi) {
    uint64_t s0=0, s1=0, s2=0, s3=0;
    size_t i = lo;
    // unroll by 32
    for (; i + 32 <= hi; i += 32) {
        s0 += p[i+0];  s0 += p[i+1];  s0 += p[i+2];  s0 += p[i+3];
        s1 += p[i+4];  s1 += p[i+5];  s1 += p[i+6];  s1 += p[i+7];
        s2 += p[i+8];  s2 += p[i+9];  s2 += p[i+10]; s2 += p[i+11];
        s3 += p[i+12]; s3 += p[i+13]; s3 += p[i+14]; s3 += p[i+15];
        s0 += p[i+16]; s0 += p[i+17]; s0 += p[i+18]; s0 += p[i+19];
        s1 += p[i+20]; s1 += p[i+21]; s1 += p[i+22]; s1 += p[i+23];
        s2 += p[i+24]; s2 += p[i+25]; s2 += p[i+26]; s2 += p[i+27];
        s3 += p[i+28]; s3 += p[i+29]; s3 += p[i+30]; s3 += p[i+31];
    }
    for (; i < hi; ++i) s0 += p[i];
    return (s0 + s1 + s2 + s3);
}

// ---------- COMPUTE helpers ----------
static uint64_t compute_block_return(uint64_t iters) {
    // vectorized FV (single-precision) version using AVX + FMA
    __m256 d0 = _mm256_set1_ps(1.0f), d1 = _mm256_set1_ps(2.0f);
    __m256 d2 = _mm256_set1_ps(3.0f), d3 = _mm256_set1_ps(4.0f);
    __m256 d4 = _mm256_set1_ps(0.5f), d5 = _mm256_set1_ps(1.5f);
    __m256 d6 = _mm256_set1_ps(2.5f), d7 = _mm256_set1_ps(3.5f);

    const __m256 a = _mm256_set1_ps(1.0000001f);
    const __m256 b = _mm256_set1_ps(0.9999997f);
    const __m256 c = _mm256_set1_ps(1.0000003f);

    for (uint64_t i = 0; i < iters; ++i) { // 256 flops per iteration
        d0 = _mm256_fmadd_ps(d0, a, d4); // 16 flops per call (8 FMAs * 2 flops each)
        d1 = _mm256_fmadd_ps(d1, b, d5);
        d2 = _mm256_fmadd_ps(d2, c, d6);
        d3 = _mm256_fmadd_ps(d3, a, d7);
        d4 = _mm256_fmadd_ps(d4, b, d0);
        d5 = _mm256_fmadd_ps(d5, c, d1);
        d6 = _mm256_fmadd_ps(d6, a, d2);
        d7 = _mm256_fmadd_ps(d7, b, d3);

        d0 = _mm256_fmadd_ps(d0, a, d4);
        d1 = _mm256_fmadd_ps(d1, b, d5);
        d2 = _mm256_fmadd_ps(d2, c, d6);
        d3 = _mm256_fmadd_ps(d3, a, d7);
        d4 = _mm256_fmadd_ps(d4, b, d0);
        d5 = _mm256_fmadd_ps(d5, c, d1);
        d6 = _mm256_fmadd_ps(d6, a, d2);
        d7 = _mm256_fmadd_ps(d7, b, d3);
    }

    __m256 sum01 = _mm256_add_ps(d0, d1);
    __m256 sum23 = _mm256_add_ps(d2, d3);
    __m256 sum45 = _mm256_add_ps(d4, d5);
    __m256 partial = _mm256_add_ps(_mm256_add_ps(sum01, sum23), sum45);

    __m128 low  = _mm256_castps256_ps128(partial);
    __m128 high = _mm256_extractf128_ps(partial, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);

    float final_sum;
    _mm_store_ss(&final_sum, sum128);
    return (uint64_t)final_sum;
}

// ---------- WRITE helpers ----------
static void write_range(uint64_t* __restrict p, size_t lo, size_t hi, uint64_t pattern, bool nt) {
    size_t i = lo;
    if (nt) {
#if defined(__AVX512F__)
        // Align to 8 qwords (64B) for streaming stores
        while ((i < hi) && (i % 8)) { p[i++] = pattern; }
        __m512i v = _mm512_set1_epi64((long long)pattern);
        for (; i + 8 <= hi; i += 8) {
            _mm512_stream_si512((__m512i*)(p + i), v);
        }
        _mm_sfence();
        for (; i < hi; ++i) p[i] = pattern;
#else
        // Fallback: scalar stores; still fence to mimic NT timing semantics
        for (; i < hi; ++i) p[i] = pattern;
        _mm_sfence();
#endif
    } else {
        for (; i < hi; ++i) p[i] = pattern;
    }
}

// ---------- Usage ----------
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -L <layers> \\\n"
        "     -Sr MiB0,MiB1,...  -rr R0,R1,...  -ci I0,I1,...  -Sw MiB0,MiB1,...  -rw R0,R1,... \\\n"
        "     [--nt-writes] [-c core] [-tR n] [-tC n] [-tW n] [--pinR list] [--pinC list] [--pinW list]\n"
        "Notes:\n"
        "  • All CSV lists must have exactly L items (unless -L omitted, legacy L=1 allowed).\n"
        "  • -tR/-tC/-tW set thread counts for READ/COMPUTE/WRITE (default 1 each).\n"
        "  • --pinR/--pinC/--pinW accept comma-separated CPU IDs to bind per kernel.\n"
        "    If list length < threads, mapping wraps (modulo). Example: --pinW 0,2,4,6\n"
        "  • You may also pin externally via `likwid-pin` or OMP_* env vars; internal --pin*\n"
        "    only applies to the kernel region it is specified for.\n",
        prog);
}

int main(int argc, char** argv) {
    // Defaults
    size_t L = 1;
    const char *sr_str=NULL, *rr_str=NULL, *ci_str=NULL, *sw_str=NULL, *rw_str=NULL;
    bool nt_writes = false;
    int core = -1;

    int tR = 1, tC = 1, tW = 1;          // per-kernel OpenMP threads
    int *pinR=NULL, *pinC=NULL, *pinW=NULL;
    size_t pinR_len=0, pinC_len=0, pinW_len=0;

    // Legacy single values fallback
    size_t legacy_read_mib = 0, legacy_write_mib = 0;
    int legacy_read_reps = 1, legacy_write_reps = 1;
    unsigned long long legacy_comp_iters = 0;

    // Parse args
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "-L") && a+1 < argc) {
            L = (size_t)strtoull(argv[++a], NULL, 10);
            if (L == 0) { fprintf(stderr, "Error: -L must be >= 1\n"); return 1; }
        } else if (!strcmp(argv[a], "-Sr") && a+1 < argc) { sr_str = argv[++a];
        } else if (!strcmp(argv[a], "-rr") && a+1 < argc) { rr_str = argv[++a];
        } else if (!strcmp(argv[a], "-ci") && a+1 < argc) { ci_str = argv[++a];
        } else if (!strcmp(argv[a], "-Sw") && a+1 < argc) { sw_str = argv[++a];
        } else if (!strcmp(argv[a], "-rw") && a+1 < argc) { rw_str = argv[++a];
        } else if (!strcmp(argv[a], "--nt-writes")) { nt_writes = true;
        } else if (!strcmp(argv[a], "-c") && a+1 < argc) { core = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "-tR") && a+1 < argc) { tR = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "-tC") && a+1 < argc) { tC = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "-tW") && a+1 < argc) { tW = atoi(argv[++a]);
        } else if (!strcmp(argv[a], "--pinR") && a+1 < argc) { pinR = parse_csv_int_dynamic(argv[++a], &pinR_len);
        } else if (!strcmp(argv[a], "--pinC") && a+1 < argc) { pinC = parse_csv_int_dynamic(argv[++a], &pinC_len);
        } else if (!strcmp(argv[a], "--pinW") && a+1 < argc) { pinW = parse_csv_int_dynamic(argv[++a], &pinW_len);
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (tR < 1) tR = 1; 
    if (tC < 1) tC = 1; 
    if (tW < 1) tW = 1;

    if (pin_to_core(core) != 0) {
        fprintf(stderr, "Warning: process-level pinning failed, continuing unpinned.\n");
    }

    // Allocate per-layer arrays
    size_t *read_mib  = NULL, *write_mib = NULL;
    int    *read_reps = NULL, *write_reps = NULL;
    unsigned long long *comp_iters = NULL;

    if (L == 1 && (!sr_str && !rr_str && !ci_str && !sw_str && !rw_str)) {
        // legacy single-layer: all default zeros except reps=1
    } else if (L == 1) {
        read_mib    = (size_t*)calloc(1, sizeof(size_t));
        write_mib   = (size_t*)calloc(1, sizeof(size_t));
        read_reps   = (int*)calloc(1, sizeof(int));
        write_reps  = (int*)calloc(1, sizeof(int));
        comp_iters  = (unsigned long long*)calloc(1, sizeof(unsigned long long));
        unsigned long long tmp;
        if (sr_str) { if (parse_csv_ull_exact(sr_str, 1, &tmp)) { fprintf(stderr, "Error parsing -Sr\n"); return 1; } read_mib[0] = (size_t)tmp; }
        if (rr_str) { if (parse_csv_ull_exact(rr_str, 1, &tmp)) { fprintf(stderr, "Error parsing -rr\n"); return 1; } read_reps[0] = (int)tmp; }
        if (ci_str) { if (parse_csv_ull_exact(ci_str, 1, &tmp)) { fprintf(stderr, "Error parsing -ci\n"); return 1; } comp_iters[0] = tmp; }
        if (sw_str) { if (parse_csv_ull_exact(sw_str, 1, &tmp)) { fprintf(stderr, "Error parsing -Sw\n"); return 1; } write_mib[0] = (size_t)tmp; }
        if (rw_str) { if (parse_csv_ull_exact(rw_str, 1, &tmp)) { fprintf(stderr, "Error parsing -rw\n"); return 1; } write_reps[0] = (int)tmp; }
        if (!sr_str) read_mib[0] = 0;
        if (!rr_str) read_reps[0] = 1;
        if (!ci_str) comp_iters[0] = 0;
        if (!sw_str) write_mib[0] = 0;
        if (!rw_str) write_reps[0] = 1;
    } else {
        if (!sr_str || !rr_str || !ci_str || !sw_str || !rw_str) {
            fprintf(stderr, "Error: For L=%zu, provide -Sr,-rr,-ci,-Sw,-rw (each with %zu items)\n", L, L);
            return 1;
        }
        read_mib   = (size_t*)malloc(L * sizeof(size_t));
        write_mib  = (size_t*)malloc(L * sizeof(size_t));
        read_reps  = (int*)malloc(L * sizeof(int));
        write_reps = (int*)malloc(L * sizeof(int));
        comp_iters = (unsigned long long*)malloc(L * sizeof(unsigned long long));

        unsigned long long *buf = (unsigned long long*)malloc(L * sizeof(unsigned long long));
        if (parse_csv_ull_exact(sr_str, L, buf)) { fprintf(stderr, "Error: -Sr must have %zu items\n", L); return 1; }
        for (size_t i=0;i<L;++i) read_mib[i] = (size_t)buf[i];

        if (parse_csv_ull_exact(rr_str, L, buf)) { fprintf(stderr, "Error: -rr must have %zu items\n", L); return 1; }
        for (size_t i=0;i<L;++i) read_reps[i] = (int)buf[i];

        if (parse_csv_ull_exact(ci_str, L, buf)) { fprintf(stderr, "Error: -ci must have %zu items\n", L); return 1; }
        for (size_t i=0;i<L;++i) comp_iters[i] = buf[i];

        if (parse_csv_ull_exact(sw_str, L, buf)) { fprintf(stderr, "Error: -Sw must have %zu items\n", L); return 1; }
        for (size_t i=0;i<L;++i) write_mib[i] = (size_t)buf[i];

        if (parse_csv_ull_exact(rw_str, L, buf)) { fprintf(stderr, "Error: -rw must have %zu items\n", L); return 1; }
        for (size_t i=0;i<L;++i) write_reps[i] = (int)buf[i];

        free(buf);
    }

    if (L == 1 && read_mib == NULL) {
        read_mib   = (size_t*)calloc(1, sizeof(size_t));
        write_mib  = (size_t*)calloc(1, sizeof(size_t));
        read_reps  = (int*)calloc(1, sizeof(int));
        write_reps = (int*)calloc(1, sizeof(int));
        comp_iters = (unsigned long long*)calloc(1, sizeof(unsigned long long));
        read_mib[0]   = legacy_read_mib;
        write_mib[0]  = legacy_write_mib;
        read_reps[0]  = legacy_read_reps;
        write_reps[0] = legacy_write_reps;
        comp_iters[0] = legacy_comp_iters;
    }

    // Allocate largest buffer once
    size_t max_mib = 0;
    for (size_t i = 0; i < L; ++i) {
        if (read_mib[i]  > max_mib) max_mib = read_mib[i];
        if (write_mib[i] > max_mib) max_mib = write_mib[i];
    }

    uint64_t* buf = NULL;
    size_t bytes = 0, n_qwords = 0;
    if (max_mib > 0) {
        bytes = (size_t)max_mib * 1024ull * 1024ull;
        buf = (uint64_t*)alloc_aligned(bytes);
        if (!buf) return 2;
        n_qwords = bytes / sizeof(uint64_t);
        for (size_t i = 0; i < n_qwords; i += 4096/8) buf[i] = (uint64_t)i;
    }

    // Init LIKWID
    LIKWID_MARKER_INIT;

    // Register marker names
    for (size_t k = 0; k < L; ++k) {
        char name[32];
        snprintf(name, sizeof(name), "read_%zu", k);    LIKWID_MARKER_REGISTER(name);
        snprintf(name, sizeof(name), "compute_%zu", k); LIKWID_MARKER_REGISTER(name);
        snprintf(name, sizeof(name), "write_%zu", k);   LIKWID_MARKER_REGISTER(name);
    }

#ifdef _OPENMP
    // Pre-initialize LIKWID threads (so first region doesn't miss init)
    int maxThreads = tR; if (tC > maxThreads) maxThreads = tC; if (tW > maxThreads) maxThreads = tW;
    if (maxThreads < 1) maxThreads = 1;
    #pragma omp parallel num_threads(maxThreads)
    {
        LIKWID_MARKER_THREADINIT;
    }
#endif

    printf("Layers: %zu  (global nt-writes=%s)  Threads: R=%d C=%d W=%d\n",
           L, nt_writes ? "on" : "off", tR, tC, tW);

    uint64_t total_start = nsec_now();
    double kernel_total_time_s = 0.0;

    for (size_t k = 0; k < L; ++k) {
        const size_t r_mib = read_mib[k];
        const int    r_rep = read_reps[k];
        const unsigned long long ci = comp_iters[k];
        const size_t w_mib = write_mib[k];
        const int    w_rep = write_reps[k];

        // ---------------- READ ----------------
        if (r_mib > 0 && r_rep > 0) {
            size_t r_bytes = (size_t)r_mib * 1024ull * 1024ull;
            size_t nQ = r_bytes / sizeof(uint64_t);
            char marker[32]; snprintf(marker, sizeof(marker), "read_%zu", k);

            uint64_t t0 = nsec_now();
            for (int r = 0; r < r_rep; ++r) {
                #pragma omp parallel num_threads(/* per-kernel */ (tR<1?1:tR))
                {
                    LIKWID_MARKER_START(marker);
                    // Optional per-thread pinning for READ
                    if (pinR && pinR_len) {
                        int tid = omp_get_thread_num();

                        int core_id = pinR[ (size_t)tid % pinR_len ];
                        (void)pin_to_core(core_id);
                    }
                    int tid = omp_get_thread_num();
                    int T   = omp_get_num_threads();

                    size_t chunk = (nQ + (size_t)T - 1) / (size_t)T;
                    size_t lo = (size_t)tid * chunk;
                    size_t hi = lo + chunk; if (hi > nQ) hi = nQ;
                    uint64_t local = read_sum_range(buf, lo, hi);

                    LIKWID_MARKER_STOP(marker);

                    #pragma omp atomic
                    g_sink += local;
                }

            }
            uint64_t t1 = nsec_now();
            double dt_s = (t1 - t0) / 1e9;
            kernel_total_time_s += dt_s;
            double MB   = ((double)r_bytes * r_rep) / 1.0e6;
            double MiB  = ((double)r_bytes * r_rep) / 1024.0 / 1024.0;
            printf("[L%zu READ]  bytes=%.0f MB=%.3f MiB=%.3f time=%.6f s  BW=%.1f MB/s (%.1f MiB/s)\n",
                   k, (double)r_bytes * r_rep, MB, MiB, dt_s, MB/dt_s, MiB/dt_s);
        } else {
            printf("[L%zu READ]  skipped\n", k);
        }

        // ---------------- COMPUTE ----------------
        if (ci > 0) {
            char marker[32]; snprintf(marker, sizeof(marker), "compute_%zu", k);
            uint64_t t0 = nsec_now();
            #ifdef _OPENMP
            #pragma omp parallel num_threads((tC<1?1:tC))
            {
                LIKWID_MARKER_START(marker);
                if (pinC && pinC_len) {
                #ifdef _OPENMP
                    int tid = omp_get_thread_num();
                #else
                    int tid = 0;
                #endif
                    int core_id = pinC[ (size_t)tid % pinC_len ];
                    (void)pin_to_core(core_id);
                }
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                int T   = omp_get_num_threads();
                #else
                int tid = 0, T = 1;
                #endif
                uint64_t base = ci / (uint64_t)T;
                uint64_t rem  = ci % (uint64_t)T;
                uint64_t localIters = base + ((uint64_t)tid < rem ? 1 : 0);
                uint64_t local = compute_block_return(localIters);

                LIKWID_MARKER_STOP(marker);
                #pragma omp atomic
                g_sink += local;
            }
        #else
            LIKWID_MARKER_START(marker);
            uint64_t local = compute_block_return(ci);
            g_sink += local;
            LIKWID_MARKER_STOP(marker);
        #endif
            uint64_t t1 = nsec_now();
            double dt_s = (t1 - t0) / 1e9;
            kernel_total_time_s += dt_s;
            printf("[L%zu COMP]  iters=%llu time=%.6f s\n",
                   k, (unsigned long long)ci, dt_s);
        } else {
            printf("[L%zu COMP]  skipped\n", k);
        }

        // ---------------- WRITE ----------------
        if (w_mib > 0 && w_rep > 0) {
            size_t w_bytes = (size_t)w_mib * 1024ull * 1024ull;
            size_t nQ = w_bytes / sizeof(uint64_t);

            // // remove first-touch faults for the region we will write
            // for (size_t i = 0; i < nQ; i += 4096/8) buf[i] = 0;

            const uint64_t pattern0 = 0xA5A5A5A5DEADBEEFull;
            char marker[32]; snprintf(marker, sizeof(marker), "write_%zu", k);

            uint64_t t0 = nsec_now();


            for (int r = 0; r < w_rep; ++r) {
            #ifdef _OPENMP
                #pragma omp parallel num_threads((tW<1?1:tW))
                {
                    LIKWID_MARKER_START(marker);
                    if (pinW && pinW_len) {
                #ifdef _OPENMP
                        int tid = omp_get_thread_num();
                #else
                        int tid = 0;
                #endif
                        int core_id = pinW[ (size_t)tid % pinW_len ];
                        (void)pin_to_core(core_id);
                    }
                #ifdef _OPENMP
                    int tid = omp_get_thread_num();
                    int T   = omp_get_num_threads();
                #else
                    int tid = 0, T = 1;
                #endif
                    size_t chunk = (nQ + (size_t)T - 1) / (size_t)T;
                    size_t lo = (size_t)tid * chunk;
                    size_t hi = lo + chunk; if (hi > nQ) hi = nQ;
                    write_range(buf, lo, hi, pattern0 + (uint64_t)r, nt_writes);

                    // nothing to add; writes already produce traffic
                    LIKWID_MARKER_STOP(marker);
                }
            #else
                LIKWID_MARKER_START(marker);
                write_range(buf, 0, nQ, pattern0 + (uint64_t)r, nt_writes);
                LIKWID_MARKER_STOP(marker);
            #endif
            }
            uint64_t t1 = nsec_now();

            double dt_s = (t1 - t0) / 1e9;
            kernel_total_time_s += dt_s;
            double MB   = ((double)w_bytes * w_rep) / 1.0e6;
            double MiB  = ((double)w_bytes * w_rep) / 1024.0 / 1024.0;
            printf("[L%zu WRITE%s] bytes=%.0f MB=%.3f MiB=%.3f time=%.6f s  BW=%.1f MB/s (%.1f MiB/s)\n",
                   k, nt_writes ? " NT" : "", (double)w_bytes * w_rep, MB, MiB, dt_s, MB/dt_s, MiB/dt_s);
        } else {
            printf("[L%zu WRITE] skipped\n", k);
        }
    }

    uint64_t total_end = nsec_now();
    double total_dt_s = (total_end - total_start) / 1e9;
    printf("Total time for all layers: %.6f s\n", total_dt_s);
    printf("Total kernel time (sum of all kernels): %.6f s\n", kernel_total_time_s);

    LIKWID_MARKER_CLOSE;

    free((void*)buf);
    free(read_mib); free(write_mib);
    free(read_reps); free(write_reps);
    free(comp_iters);
    free(pinR); free(pinC); free(pinW);

    printf("n_qwords=%zu\n", n_qwords);
    printf("sink=%llu\n", (unsigned long long)g_sink);
    return 0;
}
