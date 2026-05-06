/* QC_test_GHZ.c
 *
 * GHZ statevector benchmark/audit harness for QuEST, designed for HPC vendor
 * acceptance testing of distributed (MPI ± GPU ± OpenMP) simulation.  Vendor-
 * neutral: hardware introspection uses only POSIX APIs (sysconf, sysctlbyname,
 * /proc/cpuinfo, gethostname, uname).  No vendor-specific tooling is invoked.
 *
 * Behaviour
 *   - Reads OpenQASM 2.0 (only h, cx are simulated; comments / include / creg /
 *     measure lines are ignored).  The QASM file is NOT modified or transpiled.
 *   - Simulates with double-precision complex (qcomp = complex128).
 *   - Each MPI rank dumps its OWN local slice of the statevector to
 *         <output_dir>/statevector.rank<RRRR>.txt.gz
 *     in plain text "<global_index> <Re> <Im>" with %+.17e precision.
 *   - On every rank the same forward pass also accumulates: max |amp - ref|,
 *     total prob, P(|0..0>), P(|1..1>), and #non-zero amps.  These are
 *     reduced to rank 0 with MPI collectives, and rank 0 writes:
 *         <output_dir>/log.txt
 *         <output_dir>/MANIFEST.sha256
 *         <output_dir>/evidence/{QC_test_GHZ.c, QC_test_GHZ, <input.qasm>}
 *         <output_dir>/evidence/hosts.txt        (multi-rank only)
 *   - The reference is the analytical GHZ state:
 *         amp[0] = amp[2^N - 1] = 1/sqrt(2)   (all other amps = 0).
 *     Acceptance:  max |amp - ref|  <=  TOL  (default 1e-8).
 *
 * Vendors should additionally attach the output of their own preferred
 * hardware-listing tool (lscpu, nvidia-smi -L, rocm-smi --showproductname,
 * intel_gpu_top -L, etc.) alongside this audit package.  Such tools are
 * deliberately NOT invoked from this program in order to keep the harness
 * agnostic of any specific HPC vendor.
 *
 * Build
 *   See README.md for the full matrix.  Three typical lines:
 *     # CPU + OpenMP, single node:
 *     gcc -O3 -std=c11   QC_test_GHZ.c   -I$Q/quest/include -I$Q -I$Q/build \
 *         -L$Q/build -lQuEST -Wl,-rpath,$Q/build -lm  -o QC_test_GHZ
 *     # CPU + OpenMP + MPI, multi-node:
 *     mpicc -O3 -std=c11 -DUSE_MPI QC_test_GHZ.c   ...same flags...  -o QC_test_GHZ
 *     # GPU + MPI:  identical user-side build; QuEST itself is configured
 *     #   with -DENABLE_DISTRIBUTION=ON plus the vendor's GPU back-end
 *     #   (e.g. -DENABLE_CUDA=ON or -DENABLE_HIP=ON).  See README.md.
 *
 * Run
 *     # single-rank smoke test:
 *     ./QC_test_GHZ qasm/ghz_26.qasm  out_smoke
 *     # multi-rank, e.g. 32 MPI ranks across nodes:
 *     mpirun -np 32 ./QC_test_GHZ qasm/ghz_38.qasm  out_38
 */

#include "quest.h"
#ifdef USE_MPI
#  include <mpi.h>
#endif
#include <complex.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#if defined(__APPLE__)
#  include <sys/sysctl.h>
#endif

#define BATCH        (1u << 16)
#define LINE_BUDGET  96
#define TOL          1e-8
#define HOSTLEN      256

/* Pre-formatted line tail for amp = 0+0i (saves two %.17e formats per line). */
static const char Z_TAIL[] =
    " +0.00000000000000000e+00 +0.00000000000000000e+00\n";

typedef struct {
    Qureg  q;
    int    nQ;
    long   nH, nCX;
    double tInit, tCirc, tDump;
    /* per-rank stats (reduced to rank 0 in main) */
    double maxErr;   qindex maxErrIdx;
    double pTot, p0, pLast;
    qcomp  amp0, ampLast;        /* set only on the rank that owns idx 0 / last */
    long   nNonZero;
} Audit;

typedef struct {
    char   hostname[HOSTLEN];
    char   os[256];          /* "<sysname> <release> <machine>" */
    char   cpuModel[256];
    long   cpuLogical;
    long   cpuMemBytes;
} HwInfo;

/* ---------------- generic helpers ---------------- */

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static char* trim(char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char* e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}

static const char* backend_str(int omp, int gpu, int mpi) {
    if (gpu)        return mpi ? "GPU+MPI" : "GPU";
    if (omp && mpi) return "CPU+OMP+MPI";
    if (omp)        return "CPU+OMP";
    if (mpi)        return "CPU+MPI";
    return "CPU";
}

static void format_bytes(double b, char* out, size_t outSz) {
    static const char* u[] = {"B","KiB","MiB","GiB","TiB","PiB"};
    int i = 0;
    while (b >= 1024.0 && i < 5) { b /= 1024.0; i++; }
    snprintf(out, outSz, "%.1f %s", b, u[i]);
}

static int host_cmp(const void* a, const void* b) {
    return strncmp((const char*)a, (const char*)b, HOSTLEN);
}

/* ---------------- hardware introspection (POSIX-only, vendor-neutral) ---------------- */

static void detect_hw(HwInfo* h) {
    if (gethostname(h->hostname, sizeof h->hostname) != 0)
        snprintf(h->hostname, sizeof h->hostname, "(unknown)");
    h->hostname[HOSTLEN - 1] = 0;

    struct utsname un;
    if (uname(&un) == 0)
        snprintf(h->os, sizeof h->os, "%s %s %s", un.sysname, un.release, un.machine);
    else
        snprintf(h->os, sizeof h->os, "(unknown)");

    long onln = sysconf(_SC_NPROCESSORS_ONLN);
    h->cpuLogical = (onln > 0) ? onln : 0;

    long pages = sysconf(_SC_PHYS_PAGES);
    long pgsz  = sysconf(_SC_PAGESIZE);
    h->cpuMemBytes = (pages > 0 && pgsz > 0) ? pages * pgsz : 0;

    snprintf(h->cpuModel, sizeof h->cpuModel, "(unknown)");

#if defined(__APPLE__)
    size_t sz = sizeof h->cpuModel;
    sysctlbyname("machdep.cpu.brand_string", h->cpuModel, &sz, NULL, 0);
#elif defined(__linux__)
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof line, f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* colon = strchr(line, ':');
                if (colon) {
                    char* val = colon + 1;
                    while (*val == ' ' || *val == '\t') val++;
                    char* nl = strchr(val, '\n'); if (nl) *nl = 0;
                    snprintf(h->cpuModel, sizeof h->cpuModel, "%s", val);
                }
                break;
            }
        }
        fclose(f);
    }
#endif
}

/* ---------------- 1. parse + apply gates ---------------- */

static void parse_and_run(const char* qasm, Audit* A) {
    FILE* f = fopen(qasm, "r"); if (!f) { perror(qasm); exit(1); }
    int ready = 0, ln = 0; char line[1024];
    A->tInit = A->tCirc = 0; A->nH = A->nCX = 0;

    while (fgets(line, sizeof line, f)) {
        ln++; char* s = trim(line);
        if (*s == 0 || strncmp(s,"//",2)==0)            continue;
        if (strncmp(s,"OPENQASM",8)==0) {
            if (!strstr(s,"2.0")) { fprintf(stderr,"line %d: only OPENQASM 2.0 supported\n",ln); exit(1); }
            continue;
        }
        if (strncmp(s,"include",7)==0 || strncmp(s,"creg",4)==0
                || strncmp(s,"measure",7)==0)           continue;

        int a, b;
        if (sscanf(s,"qreg q[%d];",&a) == 1) {
            A->nQ = a;
            double t0 = now();
            A->q = createQureg(a); initZeroState(A->q);
            A->tInit = now() - t0; ready = 1; continue;
        }
        if (!ready) { fprintf(stderr,"line %d: gate before qreg\n",ln); exit(1); }

        double t0 = now();
        if      (sscanf(s,"h q[%d];",&a) == 1)            { applyHadamard(A->q,a);          A->nH++;  }
        else if (sscanf(s,"cx q[%d],q[%d];",&a,&b) == 2)  { applyControlledPauliX(A->q,a,b); A->nCX++; }
        else { fprintf(stderr,"line %d: unsupported instruction: %s\n",ln,s); exit(1); }
        A->tCirc += now() - t0;
    }
    fclose(f);
    if (!ready) { fprintf(stderr,"no qreg in %s\n",qasm); exit(1); }
}

/* ---------------- 2. dump local slice + accumulate stats ---------------- */

static void dump_local_slice(Audit* A, const char* gzPath) {
    syncQuregFromGpu(A->q);   /* no-op when no GPU */

    qindex Ntot       = (qindex)1 << A->nQ;
    qindex localCount = A->q.numAmpsPerNode;
    qindex localStart = (qindex)A->q.rank * localCount;
    qcomp* localAmps  = A->q.cpuAmps;
    qreal  invSqrt2   = 1.0 / sqrt(2.0);

    char gzCmd[2048];
    snprintf(gzCmd, sizeof gzCmd, "gzip -1 -c > '%s'", gzPath);
    FILE* sv = popen(gzCmd, "w"); if (!sv) { perror(gzCmd); exit(1); }
    fprintf(sv,
        "# QuEST statevector dump  rank=%d/%d  numQubits=%d  startIdx=%lld  count=%lld  precision=complex128\n"
        "# columns: global_index real imag\n",
        A->q.rank, A->q.numNodes, A->nQ,
        (long long)localStart, (long long)localCount);

    char* txt = malloc((size_t)LINE_BUDGET * BATCH);
    if (!txt) { fprintf(stderr,"OOM\n"); exit(1); }

    A->maxErr = 0; A->maxErrIdx = 0;
    A->pTot = A->p0 = A->pLast = 0;
    A->nNonZero = 0;
    A->amp0 = A->ampLast = 0;

    double t0 = now();

    for (qindex base = 0; base < localCount; base += BATCH) {
        qindex chunk = (base + BATCH <= localCount) ? BATCH : (localCount - base);
        char* p = txt;
        for (qindex i = 0; i < chunk; i++) {
            qindex local = base + i;
            qindex idx   = localStart + local;
            double re = creal(localAmps[local]);
            double im = cimag(localAmps[local]);
            double mod2 = re*re + im*im;

            A->pTot += mod2;
            if (idx == 0)         { A->p0    = mod2; A->amp0    = localAmps[local]; }
            if (idx == Ntot - 1)  { A->pLast = mod2; A->ampLast = localAmps[local]; }

            double rRef = (idx == 0 || idx == Ntot - 1) ? invSqrt2 : 0.0;
            double dr = re - rRef, di = im;
            double err = sqrt(dr*dr + di*di);
            if (err > A->maxErr) { A->maxErr = err; A->maxErrIdx = idx; }

            if (re == 0.0 && im == 0.0) {
                p += snprintf(p, 24, "%lld", (long long)idx);
                memcpy(p, Z_TAIL, sizeof Z_TAIL - 1);
                p += sizeof Z_TAIL - 1;
            } else {
                A->nNonZero++;
                p += snprintf(p, LINE_BUDGET, "%lld %+.17e %+.17e\n",
                              (long long)idx, re, im);
            }
        }
        fwrite(txt, 1, (size_t)(p - txt), sv);
    }
    free(txt);
    pclose(sv);
    A->tDump = now() - t0;
}

/* ---------------- 3. write the consolidated log (rank 0 only) ---------------- */

static void write_log(const char* outDir, const Audit* A,
                      const HwInfo* hw, const char* qasm, double tTotal,
                      int numRanks, int numDistinctHosts,
                      double gMaxErr, qindex gMaxErrIdx,
                      double gPTot, double gP0, double gPLast,
                      qcomp  gAmp0, qcomp  gAmpLast,
                      long   gNNonZero, double gTDump) {
    qindex N = (qindex)1 << A->nQ;
    qreal  r = 1.0 / sqrt(2.0);
    QuESTEnv env = getQuESTEnv();
    const char* be = backend_str(A->q.isMultithreaded, A->q.isGpuAccelerated, A->q.isDistributed);

    time_t tnow = time(NULL); char tbuf[64];
    strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", localtime(&tnow));
    const char* omp = getenv("OMP_NUM_THREADS"); if (!omp) omp = "(unset)";

    char ramStr[32]; format_bytes((double)hw->cpuMemBytes, ramStr, sizeof ramStr);

    char logPath[1024];
    snprintf(logPath, sizeof logPath, "%s/log.txt", outDir);
    FILE* L = fopen(logPath, "w"); if (!L) { perror(logPath); exit(1); }
    int pass = (gMaxErr <= TOL);

    fprintf(L, "QuEST GHZ benchmark log\n-----------------------\n");
    fprintf(L, "timestamp        : %s\n", tbuf);
    fprintf(L, "input qasm       : %s\n", qasm);
    fprintf(L, "output dir       : %s\n", outDir);
    fprintf(L, "QuEST version    : %d.%d.%d\n",
            QUEST_VERSION_MAJOR, QUEST_VERSION_MINOR, QUEST_VERSION_PATCH);
    fprintf(L, "precision        : FLOAT_PRECISION=%d  qcomp=%zuB → complex128\n\n",
            FLOAT_PRECISION, sizeof(qcomp));

    fprintf(L, "hardware (rank 0)\n");
    fprintf(L, "  hostname       : %s\n", hw->hostname);
    fprintf(L, "  os             : %s\n", hw->os);
    fprintf(L, "  cpu            : %s\n", hw->cpuModel);
    fprintf(L, "  cpu logical    : %ld\n", hw->cpuLogical);
    fprintf(L, "  ram            : %s\n", ramStr);
    fprintf(L, "  OMP_NUM_THREADS: %s\n\n", omp);

    fprintf(L, "deployment\n");
    fprintf(L, "  Qureg actual   : %s   (mt=%d, gpu=%d, dist=%d)\n",
            be, A->q.isMultithreaded, A->q.isGpuAccelerated, A->q.isDistributed);
    fprintf(L, "  Env available  : omp:%s  gpu:%s  mpi:%s\n",
            env.isMultithreaded   ? "on":"off", env.isGpuAccelerated ? "on":"off",
            env.isDistributed     ? "on":"off");
    fprintf(L, "  MPI ranks      : %d\n", numRanks);
    if (numRanks > 1)
        fprintf(L, "  MPI hosts      : %d distinct  (see evidence/hosts.txt)\n", numDistinctHosts);
    fprintf(L, "\n");

    fprintf(L, "circuit          : qubits=%d  gates=%ld  (H=%ld, CX=%ld)\n",
            A->nQ, A->nH + A->nCX, A->nH, A->nCX);
    fprintf(L, "amps total       : %lld (= 2^%d)\n", (long long)N, A->nQ);
    fprintf(L, "amps per rank    : %lld\n\n", (long long)A->q.numAmpsPerNode);

    fprintf(L, "timing (sec)\n");
    fprintf(L, "  parse + alloc  : %8.3f   (rank 0)\n", A->tInit);
    fprintf(L, "  gates          : %8.3f   (rank 0)\n", A->tCirc);
    fprintf(L, "  dump (gzip -1) : %8.3f   (max across ranks)\n", gTDump);
    fprintf(L, "  total wall     : %8.3f   (rank 0 wall-clock)\n\n", tTotal);

    fprintf(L, "probability sanity (reduced across ranks)\n");
    fprintf(L, "  P(|0..0>)      : %.17f\n", gP0);
    fprintf(L, "  P(|1..1>)      : %.17f\n", gPLast);
    fprintf(L, "  total          : %.17f\n\n", gPTot);

    fprintf(L, "sample amplitudes (full vector → statevector.rank<R>.txt.gz)\n");
    fprintf(L, "  amp[0]              = %+.17e %+.17e i\n",
            creal(gAmp0), cimag(gAmp0));
    fprintf(L, "  amp[%lld] = %+.17e %+.17e i\n",
            (long long)(N - 1), creal(gAmpLast), cimag(gAmpLast));
    fprintf(L, "  non-zero amps       : %ld   (expected 2 for GHZ)\n\n", gNNonZero);

    fprintf(L, "reference (analytical GHZ)\n");
    fprintf(L, "  amp[0] = amp[%lld] = 1/sqrt(2) = %.17f\n", (long long)(N - 1), r);
    fprintf(L, "  all other amps      = 0+0i\n\n");

    fprintf(L, "acceptance\n");
    fprintf(L, "  max |amp - ref|     = %.3e   at index %lld\n",
            gMaxErr, (long long)gMaxErrIdx);
    fprintf(L, "  tolerance           = %.0e\n", TOL);
    fprintf(L, "  result              = %s\n", pass ? "PASS" : "FAIL");
    fclose(L);
}

/* ---------------- 4. evidence + manifest (rank 0 only) ---------------- */

static void write_evidence(const char* outDir, const char* binary, const char* qasm) {
    char ev[1024]; snprintf(ev, sizeof ev, "%s/evidence", outDir);
    mkdir(ev, 0755);
    char cmd[4096];
    snprintf(cmd, sizeof cmd,
        "cp '%s' '%s/' 2>/dev/null;"
        "cp QC_test_GHZ.c '%s/' 2>/dev/null;"
        "cp '%s' '%s/' 2>/dev/null;"
        "cd '%s' && "
        "( command -v sha256sum >/dev/null && "
        "    sha256sum log.txt statevector.rank*.txt.gz evidence/* > MANIFEST.sha256 || "
        "  shasum -a 256 log.txt statevector.rank*.txt.gz evidence/* > MANIFEST.sha256 )",
        binary, ev, ev, qasm, ev, outDir);
    int rc = system(cmd); (void) rc;
}

/* ---------------- main ---------------- */

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.qasm> <output_dir>\n", argv[0]);
        return 2;
    }
    if (sizeof(qcomp) != 16) {
        fprintf(stderr, "FATAL: qcomp != 16B (got %zu) — not complex128\n", sizeof(qcomp));
        return 1;
    }
    const char* qasm   = argv[1];
    const char* outDir = argv[2];

    initQuESTEnv();
    QuESTEnv env = getQuESTEnv();
    int rank   = env.rank;
    int isRoot = (rank == 0);

    if (isRoot) mkdir(outDir, 0755);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);   /* ensure outDir exists on every rank */
#endif

    double tStart = now();

    Audit  A  = {0};
    HwInfo hw = {0};
    detect_hw(&hw);

    parse_and_run(qasm, &A);

    char svPath[1024];
    snprintf(svPath, sizeof svPath, "%s/statevector.rank%04d.txt.gz", outDir, rank);
    dump_local_slice(&A, svPath);

    /* ---- reduce per-rank stats to rank 0 ---- */
    double gMaxErr     = A.maxErr;
    qindex gMaxErrIdx  = A.maxErrIdx;
    double gPTot       = A.pTot;
    double gP0         = A.p0;
    double gPLast      = A.pLast;
    long   gNNonZero   = A.nNonZero;
    double gTDump      = A.tDump;
    qcomp  gAmp0       = A.amp0;
    qcomp  gAmpLast    = A.ampLast;
    int    numDistinctHosts = 1;

#ifdef USE_MPI
    {
        struct { double err; int rank; } me = { A.maxErr, rank }, best;
        MPI_Allreduce(&me, &best, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        gMaxErr = best.err;
        long long bcastIdx = (rank == best.rank) ? (long long)A.maxErrIdx : 0;
        MPI_Bcast(&bcastIdx, 1, MPI_LONG_LONG, best.rank, MPI_COMM_WORLD);
        gMaxErrIdx = (qindex)bcastIdx;

        MPI_Reduce(&A.pTot,    &gPTot,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&A.p0,      &gP0,      1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&A.pLast,   &gPLast,   1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&A.nNonZero,&gNNonZero,1, MPI_LONG,   MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&A.tDump,   &gTDump,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double samp[4] = { creal(A.amp0), cimag(A.amp0),
                           creal(A.ampLast), cimag(A.ampLast) };
        double gsamp[4];
        MPI_Reduce(samp, gsamp, 4, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        gAmp0    = gsamp[0] + I * gsamp[1];
        gAmpLast = gsamp[2] + I * gsamp[3];

        /* Gather hostnames so rank 0 can produce hosts.txt and count nodes. */
        char* allHosts = isRoot ? malloc((size_t)env.numNodes * HOSTLEN) : NULL;
        MPI_Gather(hw.hostname, HOSTLEN, MPI_CHAR,
                   allHosts,    HOSTLEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (isRoot) {
            char ev[1024]; snprintf(ev, sizeof ev, "%s/evidence", outDir);
            mkdir(ev, 0755);
            char hostsPath[1024];
            snprintf(hostsPath, sizeof hostsPath, "%s/hosts.txt", ev);
            FILE* hf = fopen(hostsPath, "w");
            if (hf) {
                fprintf(hf, "# rank  hostname\n");
                for (int i = 0; i < env.numNodes; i++)
                    fprintf(hf, "%6d  %s\n", i, allHosts + (size_t)i * HOSTLEN);
                fclose(hf);
            }

            /* count distinct hostnames via sort+dedupe on a private copy */
            char* sortBuf = malloc((size_t)env.numNodes * HOSTLEN);
            memcpy(sortBuf, allHosts, (size_t)env.numNodes * HOSTLEN);
            qsort(sortBuf, env.numNodes, HOSTLEN, host_cmp);
            numDistinctHosts = (env.numNodes > 0) ? 1 : 0;
            for (int i = 1; i < env.numNodes; i++)
                if (strncmp(sortBuf + (size_t)i*HOSTLEN,
                            sortBuf + (size_t)(i-1)*HOSTLEN, HOSTLEN) != 0)
                    numDistinctHosts++;
            free(sortBuf);
            free(allHosts);
        }
    }
#endif

    double tTotal = now() - tStart;

    if (isRoot) {
        write_log(outDir, &A, &hw, qasm, tTotal,
                  env.numNodes, numDistinctHosts,
                  gMaxErr, gMaxErrIdx,
                  gPTot, gP0, gPLast,
                  gAmp0, gAmpLast,
                  gNNonZero, gTDump);
        write_evidence(outDir, argv[0], qasm);

        int pass = (gMaxErr <= TOL);
        printf("output  : %s\n", outDir);
        printf("ranks   : %d\n", env.numNodes);
        if (env.numNodes > 1) printf("hosts   : %d distinct\n", numDistinctHosts);
        printf("backend : %s\n",
               backend_str(A.q.isMultithreaded, A.q.isGpuAccelerated, A.q.isDistributed));
        printf("max err : %.3e\n", gMaxErr);
        printf("result  : %s\n", pass ? "PASS" : "FAIL");
    }

    destroyQureg(A.q);
    finalizeQuESTEnv();   /* calls MPI_Finalize internally when MPI is enabled */

    return (gMaxErr <= TOL) ? 0 : 1;
}
