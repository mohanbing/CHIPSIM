likwid-bench -t peakflops_avx -i 50000000 -w C0:32kB:1 & likwid-bench -t load  -w C0:50GB:1
wait

# gcc -O3  -march=native -mfma -mavx512f -mavx512vl -mavx512bw  -s  -DLIKWID_PERFMON fused_bench.c -I$LIKWID_INC -L$LIKWID_LIB -llikwid -o fused_bench
# gcc -O3 -DLIKWID_PERFMON mybench.c -I$LIKWID_INC -L$LIKWID_LIB -llikwid -o mybench

# likwid-perfctr -C 0 -g MEMREAD -m ./fused_bench -i 1 -S 5000 -r 1
# likwid-perfctr -C 0 -g MEMREAD -m ./mybench
# likwid-perfctr -C 0 -g FLOPS_DP -m -- ./fused_bench -i 20000 -S 500 -r 1 



# MAIN COMMANDS TO RUN 

# sudo cpupower frequency-set -g powersave

# Compile code:
# gcc -O3 -march=native -Wall -Wextra -std=c11 fused_bench_rcw_layers_omp.c -o fused_bench_rcw_layers_omp -DLIKWID_PERFMON -llikwid -fopenmp

# Optioal code to run with likwid-perfctr
# likwid-perfctr -c 0-7 -g MEMWRITE -m ./fused_bench_rcw_layers_omp -L 3 -Sr 4096,4096,4096  -rr 2,2,2 -ci 50000000,25000000,5000000 -Sw 4096,4096,4096     -rw 1,1,1  --nt-writes  -tR 2 -tC 2 -tW 2 --pinR 0,1 --pinC 0,1 --pinW 0,1 > log1

# Example command to run 3 layers:
./fused_bench_rcw_layers_omp -L 3 \
            -Sr 4096,4096,4096  -rr 1,1,1 \
            -ci 50000000,25000000,5000000 \
            -Sw 4096,4096,4096 -rw 1,1,1  --nt-writes \
            -tR 4 -tC 2 -tW 2 --pinR 0,1,2,3,4 --pinC 0,1 --pinW 0,1