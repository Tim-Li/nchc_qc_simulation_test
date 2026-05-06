/* QC_test_GHZ.c — GHZ statevector benchmark for QuEST.
 * See README.md for build / run / acceptance details.
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

#define BATCH        (1u << 16)
#define LINE_BUDGET  96
#define TOL          1e-8

static const char Z_TAIL[] =
    " +0.00000000000000000e+00 +0.00000000000000000e+00\n";

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

/* ---- 1. parse OpenQASM 2.0 + apply gates ---- */
static Qureg parse_and_run(const char* qasm, long* nGates,
                           double* tInit, double* tCirc) {
    FILE* f = fopen(qasm, "r"); if (!f) { perror(qasm); exit(1); }
    Qureg q; int ready = 0, ln = 0; char line[1024];
    *tInit = *tCirc = 0; *nGates = 0;

    while (fgets(line, sizeof line, f)) {
        ln++; char* s = trim(line);
        if (*s == 0 || strncmp(s,"//",2)==0)            continue;
        if (strncmp(s,"OPENQASM",8)==0) {
            if (!strstr(s,"2.0")) { fprintf(stderr,"line %d: only OPENQASM 2.0\n",ln); exit(1); }
            continue;
        }
        if (strncmp(s,"include",7)==0 || strncmp(s,"creg",4)==0
                || strncmp(s,"measure",7)==0)           continue;

        int a, b;
        if (sscanf(s,"qreg q[%d];",&a) == 1) {
            double t0 = now();
            q = createQureg(a); initZeroState(q);
            *tInit = now() - t0; ready = 1; continue;
        }
        if (!ready) { fprintf(stderr,"line %d: gate before qreg\n",ln); exit(1); }

        double t0 = now();
        if      (sscanf(s,"h q[%d];",&a) == 1)            { applyHadamard(q,a);          (*nGates)++; }
        else if (sscanf(s,"cx q[%d],q[%d];",&a,&b) == 2)  { applyControlledPauliX(q,a,b);(*nGates)++; }
        else { fprintf(stderr,"line %d: unsupported: %s\n",ln,s); exit(1); }
        *tCirc += now() - t0;
    }
    fclose(f);
    if (!ready) { fprintf(stderr,"no qreg in %s\n",qasm); exit(1); }
    return q;
}

/* ---- 2. dump local slice; in the same pass, accumulate local maxErr ---- */
static void dump_and_check(Qureg q, const char* host, const char* gzPath,
                           double* tDump, double* maxErr, qindex* maxErrIdx) {
    syncQuregFromGpu(q);   /* no-op when no GPU */
    qindex Ntot       = (qindex)1 << q.numQubits;
    qindex localCount = q.numAmpsPerNode;
    qindex localStart = (qindex)q.rank * localCount;
    qcomp* amps       = q.cpuAmps;
    qreal  invSqrt2   = 1.0 / sqrt(2.0);

    char gzCmd[2048];
    snprintf(gzCmd, sizeof gzCmd, "gzip -1 -c > '%s'", gzPath);
    FILE* sv = popen(gzCmd, "w"); if (!sv) { perror(gzCmd); exit(1); }
    fprintf(sv,
        "# QuEST statevector  rank=%d/%d  host=%s  numQubits=%d  startIdx=%lld  count=%lld  precision=complex128\n"
        "# columns: global_index real imag\n",
        q.rank, q.numNodes, host, q.numQubits,
        (long long)localStart, (long long)localCount);

    char* txt = malloc((size_t)LINE_BUDGET * BATCH);
    if (!txt) { fprintf(stderr,"OOM\n"); exit(1); }

    *maxErr = 0; *maxErrIdx = 0;
    double t0 = now();

    for (qindex base = 0; base < localCount; base += BATCH) {
        qindex chunk = (base + BATCH <= localCount) ? BATCH : (localCount - base);
        qindex idxBase = localStart + base;
        char* p = txt;
        for (qindex i = 0; i < chunk; i++) {
            qindex idx  = idxBase + i;
            qcomp  z    = amps[base + i];
            double re   = creal(z), im = cimag(z);
            double rRef = (idx == 0 || idx == Ntot - 1) ? invSqrt2 : 0.0;
            double dr = re - rRef;
            double err = sqrt(dr*dr + im*im);
            if (err > *maxErr) { *maxErr = err; *maxErrIdx = idx; }

            if (re == 0.0 && im == 0.0) {
                p += snprintf(p, 24, "%lld", (long long)idx);
                memcpy(p, Z_TAIL, sizeof Z_TAIL - 1);
                p += sizeof Z_TAIL - 1;
            } else {
                p += snprintf(p, LINE_BUDGET, "%lld %+.17e %+.17e\n",
                              (long long)idx, re, im);
            }
        }
        fwrite(txt, 1, (size_t)(p - txt), sv);
    }
    free(txt); pclose(sv);
    *tDump = now() - t0;
}

/* ---- 3. log + evidence + manifest (rank 0 only) ---- */
static void write_log(const char* path, const char* qasm, const char* host,
                      Qureg q, long nGates,
                      double tInit, double tCirc, double tDump, double tTotal,
                      double maxErr, qindex maxErrIdx) {
    time_t tnow = time(NULL); char tbuf[64];
    strftime(tbuf, sizeof tbuf, "%Y-%m-%dT%H:%M:%S%z", localtime(&tnow));

    struct utsname un; uname(&un);
    long cores  = sysconf(_SC_NPROCESSORS_ONLN);
    long pgs    = sysconf(_SC_PHYS_PAGES), pgsz = sysconf(_SC_PAGESIZE);
    double ramGiB = (pgs > 0 && pgsz > 0) ? (double)pgs * pgsz / (1024.0*1024.0*1024.0) : 0;
    const char* omp = getenv("OMP_NUM_THREADS"); if (!omp) omp = "(unset)";

    qindex N  = (qindex)1 << q.numQubits;
    int    pass = (maxErr <= TOL);

    FILE* L = fopen(path, "w"); if (!L) { perror(path); exit(1); }
    fprintf(L, "timestamp        : %s\n", tbuf);
    fprintf(L, "result           : %s\n\n", pass ? "PASS" : "FAIL");

    fprintf(L, "[hardware]\n");
    fprintf(L, "  hostname       : %s\n", host);
    fprintf(L, "  os             : %s %s %s\n", un.sysname, un.release, un.machine);
    fprintf(L, "  logical cores  : %ld\n", cores);
    fprintf(L, "  ram            : %.1f GiB\n", ramGiB);
    fprintf(L, "  OMP_NUM_THREADS: %s\n", omp);
    fprintf(L, "  backend        : %s\n",
            backend_str(q.isMultithreaded, q.isGpuAccelerated, q.isDistributed));
    fprintf(L, "  MPI ranks      : %d\n\n", q.numNodes);

    fprintf(L, "[timing (sec)]\n");
    fprintf(L, "  parse + alloc  : %8.3f\n", tInit);
    fprintf(L, "  gates          : %8.3f\n", tCirc);
    fprintf(L, "  dump (gzip -1) : %8.3f\n", tDump);
    fprintf(L, "  total wall     : %8.3f\n\n", tTotal);

    fprintf(L, "[verification]\n");
    fprintf(L, "  input qasm     : %s\n", qasm);
    fprintf(L, "  QuEST version  : %d.%d.%d\n",
            QUEST_VERSION_MAJOR, QUEST_VERSION_MINOR, QUEST_VERSION_PATCH);
    fprintf(L, "  precision      : complex128 (qcomp = %zu B)\n", sizeof(qcomp));
    fprintf(L, "  qubits / gates : %d / %ld\n", q.numQubits, nGates);
    fprintf(L, "  amps total     : %lld\n", (long long)N);
    fprintf(L, "  reference      : amp[0] = amp[%lld] = 1/sqrt(2), others = 0\n", (long long)(N-1));
    fprintf(L, "  max |amp - ref|: %.3e   at index %lld\n", maxErr, (long long)maxErrIdx);
    fprintf(L, "  tolerance      : %.0e\n", TOL);
    fclose(L);
}

static void write_evidence(const char* outDir, const char* binary, const char* qasm) {
    char ev[1024]; snprintf(ev, sizeof ev, "%s/evidence", outDir);
    mkdir(ev, 0755);
    char cmd[4096];
    snprintf(cmd, sizeof cmd,
        "cp '%s' '%s/' 2>/dev/null;"
        "cp QC_test_GHZ.c '%s/' 2>/dev/null;"
        "cp '%s' '%s/' 2>/dev/null;"
        "cd '%s' && "
        "( command -v sha256sum >/dev/null "
        "    && sha256sum log.txt statevector.rank*.txt.gz evidence/* > MANIFEST.sha256 "
        "    || shasum -a 256 log.txt statevector.rank*.txt.gz evidence/* > MANIFEST.sha256 )",
        binary, ev, ev, qasm, ev, outDir);
    int rc = system(cmd); (void) rc;
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr,"usage: %s <input.qasm> <output_dir>\n",argv[0]); return 2; }
    if (sizeof(qcomp) != 16) {
        fprintf(stderr,"FATAL: qcomp != 16B (got %zu) — not complex128\n", sizeof(qcomp));
        return 1;
    }
    const char* qasm   = argv[1];
    const char* outDir = argv[2];

    initQuESTEnv();
    int rank = getQuESTEnv().rank, isRoot = (rank == 0);

    char host[256] = "?"; gethostname(host, sizeof host);

    if (isRoot) mkdir(outDir, 0755);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    double tStart = now();

    long   nGates = 0;
    double tInit, tCirc, tDump, maxErr;
    qindex maxErrIdx;

    Qureg q = parse_and_run(qasm, &nGates, &tInit, &tCirc);

    char svPath[1024];
    snprintf(svPath, sizeof svPath, "%s/statevector.rank%04d.txt.gz", outDir, rank);
    dump_and_check(q, host, svPath, &tDump, &maxErr, &maxErrIdx);

    /* reduce to rank 0 */
#ifdef USE_MPI
    {
        struct { double err; int rank; } me = {maxErr, rank}, best;
        MPI_Allreduce(&me, &best, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        maxErr = best.err;
        long long idx = (rank == best.rank) ? (long long)maxErrIdx : 0;
        MPI_Bcast(&idx, 1, MPI_LONG_LONG, best.rank, MPI_COMM_WORLD);
        maxErrIdx = (qindex)idx;

        double localTDump = tDump;
        MPI_Reduce(&localTDump, &tDump, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
#endif

    double tTotal = now() - tStart;

    if (isRoot) {
        char logPath[1024]; snprintf(logPath, sizeof logPath, "%s/log.txt", outDir);
        write_log(logPath, qasm, host, q, nGates,
                  tInit, tCirc, tDump, tTotal, maxErr, maxErrIdx);
        write_evidence(outDir, argv[0], qasm);

        int pass = (maxErr <= TOL);
        printf("output  : %s\nranks   : %d\nbackend : %s\nmax err : %.3e\nresult  : %s\n",
               outDir, q.numNodes,
               backend_str(q.isMultithreaded, q.isGpuAccelerated, q.isDistributed),
               maxErr, pass ? "PASS" : "FAIL");
    }

    destroyQureg(q);
    finalizeQuESTEnv();
    return (maxErr <= TOL) ? 0 : 1;
}
