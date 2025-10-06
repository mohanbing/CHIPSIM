#!/usr/bin/env python3
# test_global_manager.py

import os
import sys
import time
import pickle
import pprint
import argparse
from datetime import datetime
from src.sim.global_manager import GlobalManager
from src.post.output_manager import OutputManager
from src.utils.config_loader import load_config



def main():
    """
    Main simulation function.
    """
    # ====================================================
    # Configuration Selection
    # ====================================================
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a chiplet simulation.')
    parser.add_argument('--config', type=str, default="config_1", 
                       help='Config name (e.g., "vit_1_input", "vit_2_input") or full path to config file')
    
    args = parser.parse_args()
    CONFIG_NAME = args.config
    
    print(f"🔧 Using config: {CONFIG_NAME}")
    
    # Load configuration from specified experiment config
    config = load_config(CONFIG_NAME)
    
    simulation_data = {}
    simulation_data['config'] = config
    simulation_data['total_start_time'] = time.time()

    simulation_data['main_simulation_duration'] = 0.0
    
    gm = None

    print(f"⚡ Starting simulation with workload: {config['workload']}")
    print(f"⚡ Using communication simulator: {config['comm_simulator']}")
    print(f"⚡ Using communication method: {config['comm_method']}")
    print(f"⚡ Model definitions: {config['model_defs']}")
    
    gm = GlobalManager(
        wl_file_name=config['workload'],
        adj_matrix_file=config['adj_matrix'],
        chiplet_mapping_file=config['chiplet_mapping'],
        model_definitions_file=config['model_defs'],
        clear_cache=config['clear_cache'],
        communication_simulator=config['comm_simulator'],
        communication_method=config['comm_method'],
        enable_dsent=config['enable_dsent'],
        bits_per_activation=config['bits_per_activation'],
        bits_per_packet=config['bits_per_packet'],
        network_operation_frequency_hz=config['network_operation_frequency_hz'],
        gem5_sim_cycles=config['gem5_sim_cycles'],
        gem5_injection_rate=config['gem5_injection_rate'],
        gem5_ticks_per_cycle=config['gem5_ticks_per_cycle'],
        gem5_deadlock_threshold=config['gem5_deadlock_threshold'],
        dsent_tech_node=config['dsent_tech_node'],
        enable_comm_cache=config['enable_comm_cache'],
        warmup_period_us=config['warmup_period_us'],
        blocking_age_threshold=config['blocking_age_threshold'],
        weight_stationary=config['weight_stationary'],
        weight_loading_strategy=config['weight_loading_strategy'],
    )
    
    main_sim_start_time = time.time()
    gm.run_simulation()
    simulation_data['main_simulation_duration'] = time.time() - main_sim_start_time
    gm.wall_clock_runtime_s = time.time() - main_sim_start_time
        
    simulation_data['gm'] = gm
    
    if simulation_data['gm'] is None:
        print("❌ GlobalManager object not initialized. Exiting.")
        sys.exit(1)

    # ====================================================
    # Output Manager Setup & Results Directory Creation
    # ====================================================
    output_manager = OutputManager(
        wl_file_name=os.path.basename(simulation_data['gm'].workload_manager.wl_file),
        adj_matrix_file=os.path.basename(simulation_data['gm'].adj_matrix_file),
        chiplet_mapping_file=os.path.basename(simulation_data['gm'].chiplet_mapping_file),
        communication_simulator=simulation_data['gm'].communication_simulator,
        communication_method=simulation_data['gm'].communication_method,
        mapping_function=simulation_data['gm'].mapping_function,
        metric_computer=None, # MetricComputer will be used in collect_results.py
        num_chiplets=simulation_data['gm'].system.num_chiplets,
    )
    simulation_data['output_manager'] = output_manager
    # This will now be _results/raw_results/YYYY.MM.DD_...
    simulation_data['raw_results_sim_specific_dir'] = simulation_data['output_manager'].create_results_directory()
    
    # ====================================================
    # Save GlobalManager State
    # ====================================================
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    # The .pkl file is saved directly into the directory returned by create_results_directory()
    # No need to create an additional "raw_results" sub-folder here as it's already part of the path.
    
    state_filename = f"{timestamp}_post_simulation_system_state.pkl"
    # Save directly into the simulation-specific raw results directory
    state_filepath = os.path.join(simulation_data['raw_results_sim_specific_dir'], state_filename)
    
    print(f"💾 Saving simulation state to: {state_filepath}...")
    try:
        with open(state_filepath, 'wb') as f:
            pickle.dump(simulation_data['gm'], f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"⚠️ WARNING: Failed to save simulation state to {state_filepath}: {e}")
        sys.exit(1) 
    
    # ====================================================
    # Move DSENT stats file to results directory
    # ====================================================
    dsent_stats_source_path = simulation_data['gm'].dsent_stats_file
    if os.path.exists(dsent_stats_source_path) and os.path.getsize(dsent_stats_source_path) > 0:
        dsent_stats_filename = os.path.basename(dsent_stats_source_path)
        dsent_stats_dest_path = os.path.join(simulation_data['raw_results_sim_specific_dir'], dsent_stats_filename)
        
        print(f"🚚 Moving DSENT stats file to: {dsent_stats_dest_path}...")
        try:
            os.rename(dsent_stats_source_path, dsent_stats_dest_path)
            # Clean up the temporary directory if it's empty
            temp_dsent_dir = os.path.dirname(dsent_stats_source_path)
            if not os.listdir(temp_dsent_dir):
                os.rmdir(temp_dsent_dir)
        except Exception as e:
            print(f"⚠️ WARNING: Failed to move DSENT stats file: {e}")
    else:
        print("ℹ️ No DSENT stats file found or file is empty, skipping move.")
    
    # Calculate total script execution duration
    total_script_duration = time.time() - simulation_data['total_start_time']
    
    print("\n" + "=" * 80)
    print("🏁 Simulation part completed successfully!")
    print(f"🏁 Main simulation runtime: {simulation_data['main_simulation_duration']:.2f} seconds")
    
    print(f"🏁 Total script runtime: {total_script_duration:.2f} seconds")
    print(f"🏁 Raw results PKL file saved to: {state_filepath}")
    # The results_dir for run_simulation is now the specific raw results path
    print(f"🏁 Simulation-specific raw results directory: {simulation_data['raw_results_sim_specific_dir']}")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 