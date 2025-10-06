## GEM5 related instructions

### To install gem5 for network simulation
```
python3 `which scons` build/Garnet_standalone/gem5.opt -j12
```


### To run standalone with random traffic
num_dirs = 100
num-cpus = 100
network = garnet
topology: dont change
mesh-rows = 10
synthetic = 
```
./build/Garnet_standalone/gem5.opt configs/example/garnet_synth_traffic.py \
                --num-cpus=16 \
                --num-dirs=16 \
                --network=garnet \
                --topology=Mesh_XY \
                --mesh-rows=4 \
                --synthetic=uniform_random \
                --injectionrate=0.1 \
                --sim-cycles=50000
```

### To extract some stats from the output
```
source 1_757_files/extract_network_stats.sh
```

### To run with trace input  (--debug-flags=NetworkTrace for debugging)
```
build/Garnet_standalone/gem5.opt configs/example/garnet_synth_traffic.py \
                --num-cpus=100 \
                --num-dirs=100 \
                --topology=Mesh_XY \
                --mesh-rows=10 \
                --sim-cycles=50000 \
                --injectionrate=0 \
                --network-trace-enable \
                --network-trace-file="debug.log.gz" \
                --network-trace-max-packets=1

python util/on-chip-network-power-area.py .  m5out /mnt/data2/ahkanani/757_project/gem5/ext/dsent/configs/garnet_router.cfg /mnt/data2/ahkanani/757_project/gem5/ext/dsent/configs/garnet_link.cfg 32 500
```

## Known Issues
 - Network respresentations that were created and not downloaded from Cimloop likely have incorrect histograms. At minimum, models trained for imagenet were used while the cifar10 dataset was applied. 
 - Caching of communication operations is likely too conservative. Currently, too much layer-specific information is considered. 