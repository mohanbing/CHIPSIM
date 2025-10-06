# Major Components

This document describes the core modules and architecture of the Chiplet System Simulator.

## Core Modules (`src/`)

| Module | Description |
|--------|-------------|
| [`sim/global_manager.py`](../src/sim/global_manager.py) | Main simulation orchestrator |
| [`sim/compute_simulator.py`](../src/sim/compute_simulator.py) | Analog compute modeling |
| [`sim/communication_simulator.py`](../src/sim/communication_simulator.py) | NoI traffic simulation |
| [`core/system.py`](../src/core/system.py) | Chiplet system representation |
| [`mapping/model_mapper.py`](../src/mapping/model_mapper.py) | DNN-to-chiplet mapping algorithm |
| [`metrics/metric_computer.py`](../src/metrics/metric_computer.py) | Performance metric calculation |
| [`post/`](../src/post/) | Post-processing and visualization |

## Architecture Overview

### Simulation Flow

```
Configuration → System Building → Workload Execution → Result Collection
      ↓                ↓                  ↓                    ↓
  Load params    Create chiplets    Compute + Comms      Save state
  Input files    Build NoI          Schedule weights     Raw metrics
```

### Key Subsystems

#### Global Manager
The main orchestrator that coordinates all simulation components:
- Loads configuration and input files
- Initializes system components
- Manages simulation timeline
- Coordinates compute and communication simulators

#### Compute Simulator
Models analog in-memory compute operations:
- Layer computation latency
- Chiplet-specific compute characteristics
- Batch processing

#### Communication Simulator
Simulates Network-on-Interposer traffic:
- Integrates gem5 Garnet
- Cycle-accurate network simulation
- Packet routing and congestion modeling

#### System Model
Represents the chiplet-based architecture:
- Chiplet specifications and capabilities
- NoI topology and connectivity
- Resource allocation and management

#### Model Mapper
Maps DNN layers to chiplets:
- Partitioning algorithm
- Load balancing
- Communication minimization

#### Metric Computer
Computes performance metrics after the simulation is complete:
- Latency and throughput
- Network utilization
- Chiplet utilization
- Energy and power estimates

## Directory Structure

```
chiplet_simulator/
├── run_simulation.py              # Main simulation entry point
├── post_simulation_processor.py   # Post-processing entry point
├── configs/                       # Configuration files
│   ├── experiments/               # Simulation configs (*.yaml)
│   └── post_processor_jobs/       # Post-processing configs
├── assets/                        # Input files
│   ├── workloads/                 # Workload CSVs
│   ├── NoI_topologies/            # Network adjacency matrices
│   ├── chiplet_specs/             # Chiplet mappings and params
│   └── DNN_models/                # Model definitions
├── src/                           # Core simulator source
│   ├── sim/                       # Simulation engines
│   ├── core/                      # System modeling
│   ├── mapping/                   # DNN mapping algorithms
│   ├── metrics/                   # Metric computation
│   ├── post/                      # Post-processing
│   └── utils/                     # Utilities
├── helpers/                       # Helper scripts
├── tests/                         # Unit and integration tests
├── integrations/                  # External tool integrations (gem5, etc.)
├── _results/                      # Simulation outputs (generated)
└── cache/                         # Cached results (generated)
```

## Related Documentation

- [Advanced Features](advanced-features.md) - Communication caching, weight strategies, pipelining
- [GEM5 Integration](gem5-integration.md) - Network simulator setup and usage
- [Helper Scripts](helper-scripts.md) - Utilities for workload and topology generation
