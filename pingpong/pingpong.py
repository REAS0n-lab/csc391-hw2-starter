#!/usr/bin/env python3
"""Two-process ping-pong benchmark in mpi4py.

The same measurement as pingpong.c, in Python. Use whichever you prefer, but
report which one produced the numbers, because the Python version carries
interpreter overhead in every round trip and that overhead is not part of the
interconnect.

    srun -n 2 --ntasks-per-node=1 python3 pingpong/pingpong.py --csv results/pingpong-py.csv

The lowercase mpi4py calls (`comm.send`) pickle their argument. The uppercase
calls (`comm.Send`) move a buffer directly. This script uses the uppercase
form, because pickling a 64 MB array would measure the pickler.
"""

import argparse
import csv
import os
import socket
import sys

import numpy as np
from mpi4py import MPI

SIZES = [8, 64, 512, 4096, 32768, 262144, 1048576, 4194304, 16777216, 67108864]
TAG_PING, TAG_PONG = 1, 2


def reps_for(nbytes):
    if nbytes <= 4096:
        return 2000
    if nbytes <= 262144:
        return 500
    if nbytes <= 4194304:
        return 100
    return 30


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=None)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--max-bytes", type=int, default=67108864)
    args = p.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if size != 2:
        if rank == 0:
            print(f"pingpong needs exactly 2 ranks, got {size}", file=sys.stderr)
        comm.Abort(1)

    myname = socket.gethostname()
    othername = comm.sendrecv(myname, dest=1 - rank, source=1 - rank)
    same_node = myname == othername

    buf = np.ones(args.max_bytes, dtype=np.uint8)

    writer = None
    fh = None
    if rank == 0:
        print(f"# mpi_library   {MPI.Get_library_version().splitlines()[0]}")
        print(f"# mpi4py        {MPI.Get_version()}")
        print(f"# rank0_host    {myname}")
        print(f"# rank1_host    {othername}")
        print("# placement     " + ("SAME NODE, this measures shared memory"
                                    if same_node else "distinct nodes"))
        print()
        print(f"{'bytes':>12} {'trial':>8} {'one_way_s':>16} {'us':>16} {'GB_per_s':>14}")
        if same_node:
            print(f"WARNING both ranks are on {myname}. HW2 needs distinct nodes.",
                  file=sys.stderr)
        if args.csv:
            fresh = not os.path.exists(args.csv)
            fh = open(args.csv, "a", newline="")
            writer = csv.writer(fh)
            if fresh:
                writer.writerow(["bytes", "trial", "rep", "one_way_seconds",
                                 "rank0_host", "rank1_host", "same_node"])

    for nbytes in SIZES:
        if nbytes > args.max_bytes:
            continue
        view = buf[:nbytes]
        reps = reps_for(nbytes)

        for trial in range(args.trials):
            for _ in range(args.warmup):
                if rank == 0:
                    comm.Send([view, MPI.BYTE], dest=1, tag=TAG_PING)
                    comm.Recv([view, MPI.BYTE], source=1, tag=TAG_PONG)
                else:
                    comm.Recv([view, MPI.BYTE], source=0, tag=TAG_PING)
                    comm.Send([view, MPI.BYTE], dest=0, tag=TAG_PONG)

            comm.Barrier()
            t0 = MPI.Wtime()
            for _ in range(reps):
                if rank == 0:
                    comm.Send([view, MPI.BYTE], dest=1, tag=TAG_PING)
                    comm.Recv([view, MPI.BYTE], source=1, tag=TAG_PONG)
                else:
                    comm.Recv([view, MPI.BYTE], source=0, tag=TAG_PING)
                    comm.Send([view, MPI.BYTE], dest=0, tag=TAG_PONG)
            elapsed = MPI.Wtime() - t0

            if rank == 0:
                one_way = elapsed / (2.0 * reps)
                print(f"{nbytes:>12} {trial:>8} {one_way:>16.9f} "
                      f"{one_way * 1e6:>16.3f} {nbytes / one_way / 1e9:>14.4f}")
                if writer:
                    writer.writerow([nbytes, trial, reps, f"{one_way:.12e}",
                                     myname, othername, int(same_node)])

    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
