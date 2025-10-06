# Chiplet System Simulator

A co-simulation framework for evaluating DNN workload execution on chiplet-based in-memory computing systems. The simulator models both computation (analog in-memory compute) and communication (Network-on-Interposer) with cycle-accurate network simulation.

## Documentation

- **[Components & Architecture](docs/components.md)** - Core modules and system architecture
- **[Advanced Features](docs/advanced-features.md)** - Communication caching, DSENT, weight strategies
- **[GEM5 Integration](docs/gem5-integration.md)** - Network simulator setup and usage
- **[Helper Scripts](docs/helper-scripts.md)** - Workload and topology generation utilities

## Quick Start

1. **Run a simulation**
   ```bash
   python3 run_simulation.py --config <config file name>
   ```

2. **Process results**
   ```bash
   python3 post_simulation_processor.py
   ```

3. **View outputs**
   - Results saved to `_results/formatted_results/<experiment_config>`
   - Metrics, plots, and visualizations included

## Simulation Workflow

```
Config YAML → Run Simulation → Post-Process Results
     ↓              ↓                    ↓
  Setup params   Execute DNN       Analyze metrics
  Input files    Compute + Comms   Generate plots
```

**Main stages:**
1. **Configuration**: Define workload, topology, chiplet mapping, and simulation parameters
2. **Simulation**: Execute workload with compute and network co-simulation
3. **Post-processing**: Analyze results, compute metrics, generate visualizations

## Input Files

All input files are located in `assets/`:

### Workload Files (`assets/workloads/*.csv`)
Defines DNN inference requests and their injection times.

**Format:**
```csv
net_idx,inject_time_us,network,num_inputs
1,0,alexnet,1
2,100,resnet50,2
```

- `net_idx`: Unique network instance ID
- `inject_time_us`: When to inject the inference request
- `network`: DNN model name (must exist in model definitions)
- `num_inputs`: Batch size for this inference

**Generate custom workloads:**
```bash
python3 helpers/helpers_GenerateWorkload.py
```
See [Helper Scripts](docs/helper-scripts.md) for details.

### Network Topology (`assets/NoI_topologies/*.csv`)
Adjacency matrix defining the Network-on-Interposer (NoI) interconnect topology.

**Format:** CSV adjacency matrix where `matrix[i][j] = 1` indicates a direct link from chiplet `i` to chiplet `j`.

**Common topologies:**
- `adj_matrix_10x10_mesh.csv`: 100-chiplet 2D mesh
- `adj_matrix_10x10_floret.csv`: Floret topology
- `adj_matrix_10x10_mesh_with_io.csv`: Mesh with I/O chiplets

**Generate custom size mesh topologies:**
```bash
python3 helpers/helpers_GenerateAdjMatrix.py
```
See [Helper Scripts](docs/helper-scripts.md) for details.

### Chiplet Mapping (`assets/chiplet_specs/*.yaml`)
Maps chiplet IDs to chiplet types and their compute specifications.

**Format:**
```yaml
1: IO
2: Accumulator
3: SharedADC
...
```

**Available chiplet types:** `IO`, `Accumulator`, `SharedADC`, `ADCless`, `RAELLA`
- Each type defined in `assets/chiplet_specs/chiplet_params.py`
- Types differ in compute capabilities, area, and power
- Not all metrics specified in chiplet_params.py are used

### Model Definitions (`assets/DNN_models/model_definitions.py`)
Python dictionary defining DNN architectures layer-by-layer.

**Structure:**
```python
MODEL_DEFINITIONS = {
    'alexnet': {
        'layers': {
            0: {
                'description': 'Conv2d layer',
                'parameters': { ... },
                'receiving_layers': [1]
            },
            ...
        }
    }
}
```

**Add new models:**
```bash
python3 helpers/create_new_model_def.py
```
See [Helper Scripts](docs/helper-scripts.md) for details.


## Configuration

Simulation configs are YAML files in `configs/experiments/`. Each config specifies all simulation parameters.

### Example Config (`configs/experiments/config_1.yaml`)

```yaml
# Input files
workload: "workload_alexnet.csv"
adj_matrix: "adj_matrix_10x10_mesh.csv"
chiplet_mapping: "mapping_100_with_io.yaml"
model_defs: "model_definitions.py"

# Simulation parameters
comm_simulator: "Garnet"              # 'Garnet' or 'Booksim'
comm_method: "pipelined"              # 'pipelined' or 'non-pipelined'
weight_stationary: true               # Weights stay on chiplets
weight_loading_strategy: "all_at_once" # or "just_in_time"

# Network parameters
bits_per_activation: 8
bits_per_packet: 128
network_operation_frequency_hz: 1000000000  # 1 GHz

# Performance options
enable_dsent: false                   # DSENT power/area analysis
enable_comm_cache: false              # Cache network simulation results
warmup_period_us: 0.0                 # Warmup before metrics collection

# gem5 parameters (advanced)
gem5_sim_cycles: 500000000
gem5_injection_rate: 0.0
gem5_ticks_per_cycle: 1000
dsent_tech_node: "32"
```

### Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `comm_simulator` | `Garnet`, `Booksim` | Only Garnet currently supported |
| `comm_method` | `pipelined`, `non-pipelined` | Communication scheduling approach |
| `weight_stationary` | `true`, `false` | Weights loaded once vs. per-inference |
| `weight_loading_strategy` | `all_at_once`, `just_in_time` | Weight loading timing |
| `enable_dsent` | `true`, `false` | Enable power estimation |

See [Advanced Features](docs/advanced-features.md) for detailed parameter descriptions and [GEM5 Integration](docs/gem5-integration.md) for network simulation setup.

## Running Simulations

### Basic Usage

```bash
python3 run_simulation.py --config config_1
```

The config name refers to files in `configs/experiments/` (e.g., `config_1` → `configs/experiments/config_1.yaml`).

### What Happens During Simulation

1. **Load configuration** and input files
2. **Build system model**: Create chiplets and NoI topology
3. **Map DNN layers** to chiplets using partitioning algorithm
4. **Execute workload**:
   - Compute simulation: Model analog compute latency
   - Communication simulation: Gem5 Garnet or Booksim for NoI
   - Weight scheduling: Manage weight transfers
5. **Save results**: Pickled state and raw metrics to `_results/`

See [Components & Architecture](docs/components.md) for detailed information about the simulation engine.

### Simulation Output

Results are saved to `_results/raw_results/<timestamp>/`:
- `simulation_state.pkl`: Complete simulation state
- `simulation_config.yaml`: Config used for the run
- Raw timing and communication logs

## Post-Processing

After simulation completes, analyze results with the post-processor:

```bash
python3 post_simulation_processor.py
```

### Configuration

Edit `POST_PROCESSOR_JOB_FILE` in `post_simulation_processor.py` to select analysis jobs:
- `default_jobs.yaml`: Standard metrics and plots
- `quick_analysis.yaml`: Fast analysis without visualizations
- `full_analysis.yaml`: Comprehensive analysis with all features

### Analysis Features

Configure per-simulation in job files:
- **Warmup filtering**: Exclude initial warmup period from metrics
- **Communication analysis**: Per-workload, per-network, or aggregate
- **Visualization**: Chiplet utilization heatmaps, network traffic
- **Plots**: Timeline plots, throughput analysis

### Output

Enhanced results added to `_results/formatted_results/<simulation_config>/`:
- `metrics_summary.txt`: Key performance metrics
- `plots/`: Timeline and communication plots
- `visualizations/`: Chiplet mapping and utilization
- `cross_sim_comparison.csv`: Multi-run comparisons

## Additional Resources

- **[Components & Architecture](docs/components.md)** - Detailed module descriptions and directory structure
- **[Advanced Features](docs/advanced-features.md)** - Communication caching, weight strategies, DSENT
- **[GEM5 Integration](docs/gem5-integration.md)** - Installation, testing, and power analysis
- **[Helper Scripts](docs/helper-scripts.md)** - Input file generation and model management

## Known Issues

- Simulator has recently been heavily refactored. Simulation output correctness is **UNVERIFIED**.