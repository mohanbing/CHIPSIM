# Advanced Features

This document covers advanced simulation features and configuration options.

## Communication Caching

Enable communication result caching to speed up repeated simulations with identical traffic patterns.

**Configuration:**
```yaml
enable_comm_cache: true
```

**How it works:**
- Network simulation results are cached based on traffic pattern hash
- Subsequent simulations with the same pattern reuse cached results
- Dramatically reduces simulation time for parameter sweeps
- Cache stored in `cache/` directory

**When to use:**
- Parameter exploration with fixed workloads
- Iterative design space exploration
- Debugging compute logic without re-simulating network

**Limitations:**
- Only caches identical traffic patterns
- Cache invalidated if topology or network parameters change
- Requires sufficient disk space for large experiments

## Weight Loading Strategies

Configure how model weights are transferred to chiplets.

### All At Once
```yaml
weight_loading_strategy: "all_at_once"
```

- Loads all model weights at simulation start
- One-time upfront communication cost
- Best for repeated inference of same model
- Higher initial latency, lower per-inference cost

### Just In Time
```yaml
weight_loading_strategy: "just_in_time"
```

- Loads weights immediately before layer execution
- Distributed communication throughout execution
- Best for mixed workloads or single inference
- Lower initial latency, higher per-inference cost

### Weight Stationary Mode
```yaml
weight_stationary: true
```

- Weights remain on chiplets between inferences
- Only loads weights once per model (not per inference)

```yaml
weight_stationary: false
```

- Weights reloaded for each inference
- Higher communication overhead

## Communication Methods

### Pipelined Communication
```yaml
comm_method: "pipelined"
```

**Behavior:**
- Overlaps execution of model layers by independent inputs
- Layer N of input A can execute while layer N+1 of input B executes
- Maximizes throughput for batched workloads
- More complex scheduling

### Non-Pipelined Communication
```yaml
comm_method: "non-pipelined"
```

**Behavior:**
- Sequential compute-then-communicate for each input
- Layer completes fully (compute + communication) before next layer starts

## DSENT Integration

Enable power and area estimation for the Network-on-Interposer using DSENT.

```yaml
enable_dsent: true
dsent_tech_node: "32"  # Technology node in nm
```

**Requirements:**
- gem5 built with DSENT integration
- DSENT configuration files in `gem5/ext/dsent/configs/`

**Output:**
- Per-router power consumption
- Link power consumption
- Total NoI power and area estimates

**Performance note:** Enabling DSENT increases simulation time.

See [GEM5 Integration](gem5-integration.md) for DSENT setup details.

## Warmup Period

Exclude initial warmup period from metric collection to get steady-state performance.

```yaml
warmup_period_us: 1000.0  # First 1000 microseconds excluded
```

**Use cases:**
- Eliminate cold-start effects
- Measure steady-state throughput

**Note:** Post-processor can also filter warmup period. See configuration documentation.

## Network Operation Frequency

Configure the NoI operating frequency:

```yaml
network_operation_frequency_hz: 1000000000  # 1 GHz
```

Affects:
- Communication latency
- Power consumption (if DSENT enabled)
- Throughput calculations

## Packet and Activation Sizing

Configure data representation and packet sizes:

```yaml
bits_per_activation: 8    # Activation data width
bits_per_packet: 128      # Network packet size
```

**bits_per_activation:**
- Affects communication volume
- Typical values: 8, 16, 32 bits

**bits_per_packet:**
- Affects packet count and network granularity
- Typical values: 64, 128, 256 bits
- Smaller packets → more routing overhead
- Larger packets → higher serialization latency

## Advanced GEM5 Parameters

Fine-tune gem5 Garnet simulation:

```yaml
gem5_sim_cycles: 500000000    # Maximum simulation cycles
gem5_injection_rate: 0.0      # For synthetic traffic (use 0.0 for trace)
gem5_ticks_per_cycle: 1000    # Time resolution
```

**gem5_sim_cycles:**
- Can be ignored. Simulations will run until complete. 

**gem5_ticks_per_cycle:**
- gem5 internal time resolution
- Do not modify! Simulator assumes 1000 ticks per cycle. 

## Related Documentation

- [Configuration Reference](../README.md#configuration) - Full parameter list
- [GEM5 Integration](gem5-integration.md) - Network simulator details
- [Post-Processing](../README.md#post-processing) - Analysis features
- [Components](components.md) - Core simulator architecture
