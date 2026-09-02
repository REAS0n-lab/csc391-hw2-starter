/*
 * Two-process ping-pong latency and bandwidth benchmark.
 *
 * Rank 0 sends a message of n bytes to rank 1, rank 1 sends it straight back,
 * and rank 0 times the round trip. Half the round-trip time is reported as
 * the one-way time, which is the quantity a linear model
 *
 *     T(n) = alpha + beta * n
 *
 * describes. alpha is latency in seconds and beta is inverse bandwidth in
 * seconds per byte.
 *
 * Build with
 *     make
 * Run with
 *     srun -n 2 --ntasks-per-node=1 ./pingpong --csv results/pingpong.csv
 *
 * The two ranks must land on different nodes. A run with both ranks on one
 * node measures shared memory, which is a different machine. The program
 * prints both hostnames so that this is checkable rather than assumed.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_PING 1
#define TAG_PONG 2

static const size_t DEFAULT_SIZES[] = {
    8, 64, 512, 4096, 32768, 262144, 1048576, 4194304, 16777216, 67108864
};
static const int DEFAULT_NSIZES =
    (int)(sizeof(DEFAULT_SIZES) / sizeof(DEFAULT_SIZES[0]));

static int reps_for(size_t nbytes)
{
    /* Small messages need many repetitions before the timer resolution stops
     * dominating. Large messages do not, and running many of them wastes the
     * allocation. */
    if (nbytes <= 4096) return 2000;
    if (nbytes <= 262144) return 500;
    if (nbytes <= 4194304) return 100;
    return 30;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "usage: %s [--csv PATH] [--trials K] [--warmup W] [--max-bytes B]\n"
            "  --csv PATH     append one row per repetition to PATH (rank 0)\n"
            "  --trials K     independent trials per message size (default 3)\n"
            "  --warmup W     untimed round trips before each trial (default 20)\n"
            "  --max-bytes B  skip sizes above B (default 67108864)\n",
            prog);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "pingpong needs exactly 2 ranks, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const char *csv_path = NULL;
    int trials = 3, warmup = 20;
    size_t max_bytes = 67108864;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--trials") && i + 1 < argc) trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-bytes") && i + 1 < argc)
            max_bytes = strtoull(argv[++i], NULL, 10);
        else { if (rank == 0) usage(argv[0]); MPI_Abort(MPI_COMM_WORLD, 2); }
    }

    char myname[MPI_MAX_PROCESSOR_NAME], othername[MPI_MAX_PROCESSOR_NAME];
    int namelen = 0;
    memset(myname, 0, sizeof(myname));
    memset(othername, 0, sizeof(othername));
    MPI_Get_processor_name(myname, &namelen);
    MPI_Sendrecv(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 99,
                 othername, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 1 - rank, 99,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int same_node = (strcmp(myname, othername) == 0);

    if (rank == 0) {
        int vlen = 0;
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        MPI_Get_library_version(version, &vlen);
        for (char *p = version; *p; p++) if (*p == '\n') { *p = '\0'; break; }

        printf("# mpi_library   %s\n", version);
        printf("# rank0_host    %s\n", myname);
        printf("# rank1_host    %s\n", othername);
        printf("# placement     %s\n",
               same_node ? "SAME NODE, this measures shared memory"
                         : "distinct nodes");
        printf("# trials        %d\n", trials);
        printf("# warmup        %d\n", warmup);
        printf("\n%12s %8s %16s %16s %14s\n",
               "bytes", "trial", "one_way_s", "us", "GB_per_s");
        fflush(stdout);
        if (same_node)
            fprintf(stderr,
                    "WARNING both ranks are on %s. HW2 needs distinct nodes.\n",
                    myname);
    }

    FILE *csv = NULL;
    if (rank == 0 && csv_path) {
        int fresh = 1;
        FILE *probe = fopen(csv_path, "r");
        if (probe) { fresh = 0; fclose(probe); }
        csv = fopen(csv_path, "a");
        if (!csv) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD, 3); }
        if (fresh)
            fprintf(csv, "bytes,trial,rep,one_way_seconds,rank0_host,"
                         "rank1_host,same_node\n");
    }

    char *buf = (char *)malloc(max_bytes);
    if (!buf) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 4); }
    memset(buf, 1, max_bytes);

    for (int s = 0; s < DEFAULT_NSIZES; s++) {
        size_t nbytes = DEFAULT_SIZES[s];
        if (nbytes > max_bytes) continue;
        int reps = reps_for(nbytes);

        for (int t = 0; t < trials; t++) {
            for (int w = 0; w < warmup; w++) {
                if (rank == 0) {
                    MPI_Send(buf, (int)nbytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
                    MPI_Recv(buf, (int)nbytes, MPI_CHAR, 1, TAG_PONG,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf, (int)nbytes, MPI_CHAR, 0, TAG_PING,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf, (int)nbytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            for (int r = 0; r < reps; r++) {
                if (rank == 0) {
                    MPI_Send(buf, (int)nbytes, MPI_CHAR, 1, TAG_PING, MPI_COMM_WORLD);
                    MPI_Recv(buf, (int)nbytes, MPI_CHAR, 1, TAG_PONG,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    MPI_Recv(buf, (int)nbytes, MPI_CHAR, 0, TAG_PING,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf, (int)nbytes, MPI_CHAR, 0, TAG_PONG, MPI_COMM_WORLD);
                }
            }
            double elapsed = MPI_Wtime() - t0;

            if (rank == 0) {
                double one_way = elapsed / (2.0 * reps);
                double gbs = (double)nbytes / one_way / 1e9;
                printf("%12zu %8d %16.9f %16.3f %14.4f\n",
                       nbytes, t, one_way, one_way * 1e6, gbs);
                fflush(stdout);
                if (csv)
                    fprintf(csv, "%zu,%d,%d,%.12e,%s,%s,%d\n",
                            nbytes, t, reps, one_way, myname, othername, same_node);
            }
        }
    }

    free(buf);
    if (csv) fclose(csv);
    MPI_Finalize();
    return 0;
}
