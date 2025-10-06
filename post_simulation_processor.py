#!/usr/bin/env python3
# post_simulation_processor.py

import os
import sys
import time
# import argparse  # No longer needed - using config constant instead
import pickle
import yaml # For loading the config file
import glob # For finding the .pkl file
import shutil
import csv
import numpy as np

from src.post.output_manager import OutputManager
from src.post.visualize_mapping import ChipletVisualizer
from src.post.simulation_plotter import SimulationPlotter
from src.metrics.metric_computer import MetricComputer
from src.metrics.metric_formatter import MetricFormatter
from src.post.cross_sim_processor import CrossSimProcessor

# Configuration for post processor job file
# Available job files in configs/post_processor_jobs/:
#   - default_jobs.yaml: Standard processing jobs
#   - quick_analysis.yaml: Fast analysis without plots/visualizations  
#   - full_analysis.yaml: Complete analysis with all features enabled
POST_PROCESSOR_JOB_FILE = "hw_comparison.yaml"  # Change this to use different job configurations

# Default values for processing if not specified in a config entry
DEFAULT_PROCESSING_CONFIG = {
    "warmup_period_us": 0.0,
    "run_wkld_agg_comm": False,
    "run_ind_comm": False,
    "run_net_agg_comm": False,
    "generate_plots": True,
    "generate_visualizations": False
}

def format_config_readable(config_item):
    """
    Format a config item in a human-readable way for error reporting.
    """
    print("📋 Configuration Details:")
    print("=" * 50)
    
    # Core simulation parameters
    print(f"  Workload Name:        {config_item.get('workload_name', 'N/A')}")
    print(f"  Communication Sim:    {config_item.get('comm_simulator', 'N/A')}")
    print(f"  Communication Method: {config_item.get('comm_method', 'N/A')}")
    print(f"  Number of Chiplets:   {config_item.get('num_chiplets', 'N/A')}")
    print(f"  Adjacency Matrix:     {config_item.get('adj_matrix_file', 'N/A')}")
    print(f"  Chiplet Mapping:      {config_item.get('chiplet_mapping_file', 'N/A')}")
    
    # Processing options
    print("\n  Processing Options:")
    print(f"    Warmup Period (μs):       {config_item.get('warmup_period_us', DEFAULT_PROCESSING_CONFIG['warmup_period_us'])}")
    print(f"    Run Workload Agg Comm:   {config_item.get('run_wkld_agg_comm', DEFAULT_PROCESSING_CONFIG['run_wkld_agg_comm'])}")
    print(f"    Run Individual Comm:     {config_item.get('run_ind_comm', DEFAULT_PROCESSING_CONFIG['run_ind_comm'])}")
    print(f"    Run Network Agg Comm:    {config_item.get('run_net_agg_comm', DEFAULT_PROCESSING_CONFIG['run_net_agg_comm'])}")
    print(f"    Generate Plots:          {config_item.get('generate_plots', DEFAULT_PROCESSING_CONFIG['generate_plots'])}")
    print(f"    Generate Visualizations: {config_item.get('generate_visualizations', DEFAULT_PROCESSING_CONFIG['generate_visualizations'])}")
    
    # Show any additional fields in the config
    additional_fields = {k: v for k, v in config_item.items() 
                        if k not in ['workload_name', 'comm_simulator', 'comm_method', 'num_chiplets', 
                                   'adj_matrix_file', 'chiplet_mapping_file', 'warmup_period_us',
                                   'run_wkld_agg_comm', 'run_ind_comm', 'run_net_agg_comm', 
                                   'generate_plots', 'generate_visualizations']}
    
    if additional_fields:
        print("\n  Additional Fields:")
        for key, value in additional_fields.items():
            print(f"    {key}: {value}")
    
    print("=" * 50)

def find_raw_results_folder(base_raw_results_dir, config_item):
    """
    Searches for raw results folders matching the criteria in config_item.
    A folder matches if its name contains all the specified search components.
    Returns a list of matching folder paths and an optional error message for critical errors.
    """
    workload_name = config_item.get("workload_name")
    comm_simulator = config_item.get("comm_simulator")
    comm_method = config_item.get("comm_method")
    adj_matrix_file = config_item.get("adj_matrix_file")
    chiplet_mapping_file = config_item.get("chiplet_mapping_file")
    num_chiplets = config_item.get("num_chiplets")

    if not all([workload_name, comm_simulator, comm_method, adj_matrix_file, chiplet_mapping_file, num_chiplets is not None]):
        return [], (f"Config item is missing one or more required fields for search: "
                    f"workload_name, comm_simulator, comm_method, adj_matrix_file, "
                    f"chiplet_mapping_file, num_chiplets. Config: {config_item}")

    # Strip extensions from filenames to match directory naming convention
    workload_base_name = os.path.splitext(workload_name)[0]
    adj_matrix_base_name = os.path.splitext(adj_matrix_file)[0]
    chiplet_mapping_base_name = os.path.splitext(chiplet_mapping_file)[0]
    
    # Construct search components from the config
    search_components = [
        workload_base_name,
        adj_matrix_base_name,
        chiplet_mapping_base_name,
        f"{num_chiplets}chiplets",
        comm_simulator,
        comm_method
    ]
    
    matching_folders = []
    if not os.path.isdir(base_raw_results_dir):
        # This is a critical error with the environment.
        return [], f"Base raw results directory not found: {base_raw_results_dir}"

    for folder_name in os.listdir(base_raw_results_dir):
        full_folder_path = os.path.join(base_raw_results_dir, folder_name)
        if os.path.isdir(full_folder_path):
            # Check if all components are present in the folder name
            if all(component in folder_name for component in search_components):
                # If searching for "pipelined", explicitly exclude "non-pipelined" folders
                if config_item.get("comm_method") == "pipelined":
                    if "non-pipelined" in folder_name:
                        continue  # Skip this folder, it's a false positive
                
                matching_folders.append(full_folder_path)
    
    # If no folders match, an empty list is returned. The caller (main) will handle this.
    # If multiple folders match, the list containing all of them is returned.
    return matching_folders, None # Return the list of matching folders (empty or not) and no error for match count issues.


def find_pkl_file(sim_specific_raw_dir):
    """Finds the .pkl file in the given directory."""
    pkl_files = glob.glob(os.path.join(sim_specific_raw_dir, "*.pkl"))

    if len(pkl_files) != 1:
        return None, f"Expected exactly 1 .pkl file in {sim_specific_raw_dir}, but found {len(pkl_files)}. Files: {pkl_files}"
    
    return pkl_files[0], None

def main():
    overall_start_time = time.time()

    # Construct path to the job configuration file
    config_file_path = os.path.join(os.getcwd(), "configs", "post_processor_jobs", POST_PROCESSOR_JOB_FILE)
    
    try:
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
            processing_configs = config_data.get('jobs', [])
    except Exception as e:
        print(f"❌ ERROR: Failed to load or parse config file {config_file_path}: {e}")
        sys.exit(1)

    if not isinstance(processing_configs, list):
        print(f"❌ ERROR: Config file {config_file_path} must contain a YAML 'jobs' list of configurations.")
        sys.exit(1)
        
    base_raw_results_dir = os.path.join(os.getcwd(), "_results", "raw_results")
    base_formatted_results_dir = os.path.join(os.getcwd(), "_results", "formatted_results")
    os.makedirs(base_formatted_results_dir, exist_ok=True)

    total_configs = len(processing_configs)
    processed_count = 0
    all_metric_computers_for_analysis = [] # To store (config_info, metric_computer_instance)

    print(f"Found {total_configs} configurations to process.")

    for i, config_item in enumerate(processing_configs):
        print(f"\n" + "="*80)
        print(f"Processing config {i+1}/{total_configs}: {config_item.get('workload_name', 'N/A')}_{config_item.get('comm_simulator', 'N/A')}_{config_item.get('comm_method', 'N/A')}")
        
        config_specific_overall_start_time = time.time() # For timing this whole config item

        # --- Get processing parameters from config_item or use defaults ---
        warmup_period = config_item.get("warmup_period_us", DEFAULT_PROCESSING_CONFIG["warmup_period_us"])
        run_wkld_agg_comm = config_item.get("run_wkld_agg_comm", DEFAULT_PROCESSING_CONFIG["run_wkld_agg_comm"])
        run_ind_comm = config_item.get("run_ind_comm", DEFAULT_PROCESSING_CONFIG["run_ind_comm"])
        run_net_agg_comm = config_item.get("run_net_agg_comm", DEFAULT_PROCESSING_CONFIG["run_net_agg_comm"])
        generate_plots = config_item.get("generate_plots", DEFAULT_PROCESSING_CONFIG["generate_plots"])
        generate_visualizations = config_item.get("generate_visualizations", DEFAULT_PROCESSING_CONFIG["generate_visualizations"])

        # --- 1. Find the raw results folder(s) ---
        list_of_sim_specific_raw_dirs, err = find_raw_results_folder(base_raw_results_dir, config_item)
        if err: # Handles critical errors like missing base dir or bad config item for search
            print(f"\n❌ CRITICAL ERROR: Issue encountered while searching for raw results for config {i+1}")
            print(f"🚨 Reason: {err}")
            print(f"\n🔧 Configuration that caused the error:")
            format_config_readable(config_item)
            sys.exit(1)

        if not list_of_sim_specific_raw_dirs:
            print(f"\n❌ ERROR: No raw results folder found for config {i+1}")
            print(f"🔍 Searched in directory: {base_raw_results_dir}")
            
            # Show what we were searching for
            workload_base_name = os.path.splitext(config_item.get('workload_name', ''))[0]
            adj_matrix_base_name = os.path.splitext(config_item.get('adj_matrix_file', ''))[0]
            chiplet_mapping_base_name = os.path.splitext(config_item.get('chiplet_mapping_file', ''))[0]
            
            search_components = [
                workload_base_name,
                adj_matrix_base_name,
                chiplet_mapping_base_name,
                f"{config_item.get('num_chiplets', 'N')}chiplets",
                config_item.get('comm_simulator', ''),
                config_item.get('comm_method', '')
            ]
            
            print(f"🎯 Expected folder to contain ALL of these components:")
            for component in search_components:
                print(f"   • {component}")
            
            # Show what folders actually exist in the directory
            if os.path.exists(base_raw_results_dir):
                existing_folders = [f for f in os.listdir(base_raw_results_dir) 
                                  if os.path.isdir(os.path.join(base_raw_results_dir, f))]
                if existing_folders:
                    print(f"\n📁 Existing folders in {base_raw_results_dir}:")
                    for folder in sorted(existing_folders)[:10]:  # Show first 10 folders
                        print(f"   • {folder}")
                    if len(existing_folders) > 10:
                        print(f"   ... and {len(existing_folders) - 10} more folders")
                else:
                    print(f"\n📁 No folders found in {base_raw_results_dir}")
            else:
                print(f"\n📁 Directory {base_raw_results_dir} does not exist")
            
            print(f"\n🔧 Failed configuration details:")
            format_config_readable(config_item)
            
            print(f"\n💡 Tips:")
            print(f"   • Check if the simulation has been run for this configuration")
            print(f"   • Verify the config parameters match the simulation that was run")
            print(f"   • Check if the raw results directory path is correct")
            
            sys.exit(1)

        print(f"ℹ️ Found {len(list_of_sim_specific_raw_dirs)} matching raw results folder(s) for config {i+1}:")
        for dir_path_idx, dir_path in enumerate(list_of_sim_specific_raw_dirs):
            print(f"  Match {dir_path_idx+1}: {dir_path}")

        num_matches_for_config = len(list_of_sim_specific_raw_dirs)
        successfully_processed_matches_for_config = 0

        # --- Determine Base Formatted Output Directory for this config item ---
        # Uses WORKLOAD_COMM_SIM_COMM_METHOD as the directory name for the config.
        config_workload_name = config_item.get("workload_name", "unknown_workload")
        config_comm_simulator = config_item.get("comm_simulator", "unknown_simulator")
        config_comm_method = config_item.get("comm_method", "unknown_method")

        # Get additional parameters for a more descriptive formatted directory name
        config_adj_matrix = config_item.get("adj_matrix_file", "unknown_adj")
        config_chiplet_mapping = config_item.get("chiplet_mapping_file", "unknown_mapping")
        config_num_chiplets = config_item.get("num_chiplets", "N")

        # Strip extensions for cleaner directory names
        config_workload_name = os.path.splitext(config_workload_name)[0]
        config_adj_matrix = os.path.splitext(config_adj_matrix)[0]
        config_chiplet_mapping = os.path.splitext(config_chiplet_mapping)[0]

        config_based_formatted_dir_name = (f"{config_workload_name}_{config_comm_simulator}_"
                                           f"{config_comm_method}_{config_adj_matrix}_"
                                           f"{config_chiplet_mapping}_{config_num_chiplets}chiplets")

        main_config_formatted_output_dir = os.path.join(base_formatted_results_dir, config_based_formatted_dir_name)
        # --- Delete old results directory if it exists ---
        if os.path.exists(main_config_formatted_output_dir):
            print(f"⚠️ Deleting old formatted results directory: {main_config_formatted_output_dir}")
            shutil.rmtree(main_config_formatted_output_dir)
        os.makedirs(main_config_formatted_output_dir, exist_ok=True)
        print(f"ℹ️ Main formatted output directory for this config: {main_config_formatted_output_dir}")

        for match_idx, sim_specific_raw_dir in enumerate(list_of_sim_specific_raw_dirs):
            print(f"\n--- Starting processing for match {match_idx+1}/{num_matches_for_config} of config {i+1} ---")
            print(f"--- Target raw directory: {sim_specific_raw_dir} ---")
            
            simulation_data = {} # Reset for each specific raw directory processing
            simulation_data['current_raw_dir'] = sim_specific_raw_dir
            simulation_data['match_specific_start_time'] = time.time()

            # --- 2. Find the .pkl file within that folder ---
            raw_results_pkl_file, err_pkl = find_pkl_file(sim_specific_raw_dir)
            if err_pkl:
                print(f"❌ ERROR: Could not find .pkl file for raw directory {sim_specific_raw_dir} (match {match_idx+1} of config {i+1}). Reason: {err_pkl}. Halting.")
                sys.exit(1)
            assert raw_results_pkl_file is not None  # Type assertion after error check
            print(f"ℹ️ Found .pkl file: {raw_results_pkl_file}")

            # --- 3. Load Simulation State ---
            gm = None
            print(f"🔄 Loading simulation results from: {raw_results_pkl_file}...")
            load_start_time = time.time()
            try:
                with open(raw_results_pkl_file, 'rb') as f:
                    gm = pickle.load(f)
                print(f"✅ Successfully loaded state from: {raw_results_pkl_file}")
            except Exception as e:
                print(f"❌ Error loading state from {raw_results_pkl_file}: {e}. Halting.")
                sys.exit(1)
            load_duration = time.time() - load_start_time
            print(f"⏱️ Loading state took {load_duration:.2f} seconds")
            
            simulation_data['gm'] = gm
            if simulation_data['gm'] is None:
                print("❌ GlobalManager object not loaded after attempting to load. Exiting.")
                sys.exit(1)
            
            # Migrate legacy attribute names if needed (for backward compatibility)
            simulation_data['gm']._migrate_legacy_attributes()

            # --- 4. Apply Warmup Period and Filter ---
            print(f"ℹ️ Applying warmup period of {warmup_period} μs for post-processing.")
            simulation_data['gm'].warmup_period_us = warmup_period
            
            print("🔄 Filtering retired models by warmup period...")
            simulation_data['gm'].filter_retired_networks_by_warmup()
            print(f"✅ Models filtered. {len(simulation_data['gm'].post_warmup_retired_models)} models remaining post-warmup.")

            # --- 5. Determine Formatted Output Directory for this specific match ---
            # Create a numbered subdirectory within the main_config_formatted_output_dir
            current_run_formatted_subdir_name = str(match_idx + 1) # Names subfolders "1", "2", etc.
            current_formatted_results_dir = os.path.join(main_config_formatted_output_dir, current_run_formatted_subdir_name)
            os.makedirs(current_formatted_results_dir, exist_ok=True)
            print(f"ℹ️ Formatted results for this match will be saved to: {current_formatted_results_dir}")
            simulation_data['results_dir'] = current_formatted_results_dir

            # --- 6. Run Supplementary Simulations ---
            simulation_data['workload_aggregate_results'] = None
            simulation_data['individual_results'] = None
            simulation_data['network_aggregate_results'] = None
            simulation_data['aggregate_sim_duration'] = 0
            simulation_data['individual_sim_duration'] = 0
            simulation_data['network_aggregate_sim_duration'] = 0

            if run_wkld_agg_comm:
                print("🔄 Running workload aggregate communication simulation...")
                agg_start_time = time.time()
                simulation_data['workload_aggregate_results'] = simulation_data['gm'].comm_simulator.simulate_workload_aggregate_communication(simulation_data['gm'].retired_mapped_models)
                simulation_data['aggregate_sim_duration'] = time.time() - agg_start_time
                print(f"⏱️ Workload aggregate simulation took {simulation_data['aggregate_sim_duration']:.2f} seconds")
            
            if run_ind_comm:
                print("🔄 Running individual layer communication simulation...")
                ind_start_time = time.time()
                simulation_data['individual_results'] = simulation_data['gm'].comm_simulator.simulate_individual_layer_communication(simulation_data['gm'].retired_mapped_models)
                simulation_data['individual_sim_duration'] = time.time() - ind_start_time
                print(f"⏱️ Individual layer simulation took {simulation_data['individual_sim_duration']:.2f} seconds")
            
            if run_net_agg_comm:
                print("🔄 Running network aggregate communication simulation...")
                net_agg_start_time = time.time()
                simulation_data['network_aggregate_results'] = simulation_data['gm'].comm_simulator.simulate_network_aggregate_communication(simulation_data['gm'].retired_mapped_models)
                simulation_data['network_aggregate_sim_duration'] = time.time() - net_agg_start_time
                print(f"⏱️ Network aggregate simulation took {simulation_data['network_aggregate_sim_duration']:.2f} seconds")
            
            # # Accum homogeneous numbers    
            # simulation_data['empty_system_individual_model_results'] = { 
            #     "alexnet": {"latency": {"total": 213.46, "compute": 14.98, "communication": 198.48}, "energy": {}},
            #     "resnet18": {"latency": {"total": 72.18, "compute": 27.48, "communication": 44.70}, "energy": {}},
            #     "resnet34": {"latency": {"total": 165.68, "compute": 57.72, "communication": 107.95}, "energy": {}},
            #     "resnet50": {"latency": {"total": 460.49, "compute": 50.47, "communication": 410.02}, "energy": {}},
            # }
            
            # Accum-Raella Heterogeneous Numbers (alternating chiplets)
            simulation_data['empty_system_individual_model_results'] = {
                "alexnet": {"latency": {"total": 255.60, "compute": 18.10, "communication": 237.49}, "energy": {}},
                "resnet18": {"latency": {"total": 72.18, "compute": 27.48, "communication": 44.70}, "energy": {}},
                "resnet34": {"latency": {"total": 178.06, "compute": 57.72, "communication": 120.33}, "energy": {}},
                "resnet50": {"latency": {"total": 500.48, "compute": 105.75, "communication": 394.73}, "energy": {}},
            }
            

            # --- 7. Compute Metrics ---
            print("⚙️ Computing all metrics...")
            metric_computer_start_time = time.time()

            # The dsent_stats.jsonl file was moved to the raw results directory by run_simulation.py.
            # We must construct the path to it within the specific raw results directory for this match.
            dsent_stats_path = os.path.join(sim_specific_raw_dir, 'dsent_stats.jsonl')

            metric_computer = MetricComputer(
                simulation_data['gm'].post_warmup_retired_models, 
                simulation_data['gm'].global_time_us,
                simulation_data['gm'].system.num_chiplets,
                dsent_stats_file_path=dsent_stats_path
            )
            simulation_data['metric_computer'] = metric_computer
            metric_computer.compute_avg_system_utilization()
            metric_computer.compute_utilization_over_time(simulation_data['gm'].time_step_us)
            simulation_data['model_summary_metrics_raw'] = metric_computer.compute_model_summary_metrics()
            print(f"Generated summary metrics for {len(simulation_data['model_summary_metrics_raw'])} models")
            metric_computer.compute_approach_comparison_metrics(
                simulation_data.get('individual_results'), 
                simulation_data['empty_system_individual_model_results']
            )

            # Compute power profile. The MetricComputer will load the stats from the file path provided.
            print("🔄 Computing power profile...")
            metric_computer.compute_power_profile(
                time_step_us=simulation_data['gm'].time_step_us
            )
            
            # Compute energy profile.
            print("🔄 Computing energy profile...")
            metric_computer.compute_energy_metrics()
            
            # metric_computer.total_simulation_time_us = simulation_data['gm'].global_time_us  # Attribute doesn't exist

            metric_computer_duration = time.time() - metric_computer_start_time
            print(f"⏱️ Metric computation took {metric_computer_duration:.2f} seconds")

            # Store MetricComputer instance and relevant info for later cross-analysis
            analysis_item_info = {
                "workload_name": config_item.get("workload_name"),
                "comm_simulator": config_item.get("comm_simulator"),
                "comm_method": config_item.get("comm_method"),
                "adj_matrix_file": os.path.basename(simulation_data['gm'].adj_matrix_file),
                "chiplet_mapping_file": os.path.basename(simulation_data['gm'].chiplet_mapping_file),
                "match_identifier": os.path.basename(sim_specific_raw_dir), # Unique identifier for the match
                "results_dir": current_formatted_results_dir
            }
            all_metric_computers_for_analysis.append({
                "config_info": analysis_item_info,
                "metric_computer": metric_computer
            })

            # --- 8. Output Manager Setup ---
            output_manager = OutputManager(
                wl_file_name=os.path.basename(simulation_data['gm'].workload_manager.wl_file),
                adj_matrix_file=os.path.basename(simulation_data['gm'].adj_matrix_file),
                chiplet_mapping_file=os.path.basename(simulation_data['gm'].chiplet_mapping_file),
                communication_simulator=simulation_data['gm'].communication_simulator,
                communication_method=simulation_data['gm'].communication_method,
                mapping_function=simulation_data['gm'].mapping_function,
                metric_computer=simulation_data['metric_computer'],
                results_dir=current_formatted_results_dir,
                num_chiplets=simulation_data['gm'].system.num_chiplets
            )
            simulation_data['output_manager'] = output_manager
                    
            # --- 9. Metric Formatting & Saving ---
            print("📝 Formatting and saving metrics...")
            formatting_start_time = time.time()
            metric_formatter = MetricFormatter(metric_computer=simulation_data['metric_computer'], 
                                               global_manager=simulation_data['gm'])
            
            formatted_model_metrics = metric_formatter.format_all_model_metrics()
            formatted_utilization_metrics = metric_formatter.format_utilization_metrics(simulation_data['gm'].time_step_us)
            formatted_comparison_metrics = metric_formatter.format_approach_comparison_metrics()
            formatted_simulation_summary = metric_formatter.format_simulation_summary(simulation_data['gm'].wall_clock_runtime_s if hasattr(simulation_data['gm'], 'wall_clock_runtime_s') else 0)
            formatted_energy_metrics = metric_formatter.format_energy_metrics()
            
            output_manager.save_formatted_metrics(formatted_model_metrics, subdirectory="formatted_model_metrics")
            output_manager.save_formatted_metrics(formatted_utilization_metrics, subdirectory="formatted_utilization_metrics")
            output_manager.save_formatted_metrics(formatted_comparison_metrics, subdirectory="formatted_comparison_metrics")
            output_manager.save_formatted_metrics(formatted_simulation_summary, subdirectory=None)
            output_manager.save_formatted_metrics(formatted_energy_metrics, subdirectory="formatted_energy_metrics")
            formatting_duration = time.time() - formatting_start_time
            print(f"⏱️ Metric formatting and saving took {formatting_duration:.2f} seconds")
            
            # Save the total power over time of each chiplet, averaged over 100-microsecond intervals.
            power_data = simulation_data['metric_computer'].chiplet_total_power_over_time
            if power_data:
                power_csv_path = os.path.join(current_formatted_results_dir, 'chiplet_power_100us_avg.csv')
                print(f"🐛 Saving 100-microsecond averaged chiplet power to {power_csv_path}")
                
                time_step_us = simulation_data['gm'].time_step_us
                
                if time_step_us > 0:
                    steps_per_100us = int(100 / time_step_us)
                    if steps_per_100us > 0:
                        # Sort chiplet IDs for consistent order
                        sorted_chiplet_ids = sorted(power_data.keys())
                        max_power = 140.0

                        with open(power_csv_path, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            for chiplet_id in sorted_chiplet_ids:
                                chiplet_name = f"chiplet_chiplet_{chiplet_id}_chiplet"
                                power_values = power_data[chiplet_id]
                                
                                # Pad the array to be a multiple of steps_per_100us
                                num_values = len(power_values)
                                num_to_pad = (steps_per_100us - num_values % steps_per_100us) % steps_per_100us
                                if num_to_pad > 0:
                                    padded_values = np.pad(power_values, (0, num_to_pad), mode='constant', constant_values=np.nan)
                                else:
                                    padded_values = power_values
                                
                                # Reshape and compute mean, ignoring NaNs
                                reshaped_values = padded_values.reshape(-1, steps_per_100us)
                                averaged_power = np.nanmean(reshaped_values, axis=1)

                                power_percentages = (averaged_power / max_power) * 100
                                writer.writerow([chiplet_name] + list(power_percentages))
                    else:
                        print("🐛 Time step is too large to average power over 100us intervals. No power data saved.")
                else:
                    print("🐛 Time step is 0 or negative. Cannot average power. No power data saved.")
            else:
                print("🐛 No power data to save.")

            # --- 10. Create Plots ---
            if generate_plots:
                print("📊 Generating plots...")
                plotting_start_time = time.time()
                plotter = SimulationPlotter(results_folder=simulation_data['results_dir'], 
                                            metric_computer=simulation_data['metric_computer'])
                plotter.plot_utilization_over_time()
                plotter.plot_approach_comparison_metrics()
                plotter.plot_power_over_time()
                plotting_duration = time.time() - plotting_start_time
                print(f"⏱️ Plot generation took {plotting_duration:.2f} seconds")
            
            # --- 11. Generate Visualizations ---
            if generate_visualizations:
                print("🎨 Generating visualizations...")
                viz_start_time = time.time()
                visualizer = ChipletVisualizer(adj_matrix_file=simulation_data['gm'].adj_matrix_file, 
                                               results_folder=simulation_data['results_dir'])
                network_viz_output_dir = os.path.join(simulation_data['results_dir'], "network_mapping_visualizations")
                os.makedirs(network_viz_output_dir, exist_ok=True)
                visualizer.visualize_network_mappings_from_data(
                    retired_mapped_models=simulation_data['gm'].post_warmup_retired_models,
                    output_dir=network_viz_output_dir
                )
                visualizer.visualize_system_state_over_time(
                    retired_mapped_models=simulation_data['gm'].post_warmup_retired_models,
                    output_dir=network_viz_output_dir
                )
                viz_duration = time.time() - viz_start_time
                print(f"⏱️ Visualization generation took {viz_duration:.2f} seconds")
            
            match_specific_duration = time.time() - simulation_data['match_specific_start_time']
            print(f"--- Finished processing match {match_idx+1}/{num_matches_for_config}. Duration: {match_specific_duration:.2f} seconds ---")
            successfully_processed_matches_for_config +=1
        
        # This point is reached only if all matches for the current config_item were processed successfully.
        processed_count +=1
        config_specific_total_duration = time.time() - config_specific_overall_start_time
        print(f"🏁 Finished all {successfully_processed_matches_for_config} match(es) for config {i+1}. Total duration for this config: {config_specific_total_duration:.2f} seconds.")

    # --- End of Loop ---
    print("\n" + "="*80)
    print("🏁 All configurations processed successfully.")
    print(f"Total configurations processed: {processed_count}")
    
    # --- Perform Cross-Simulation Analysis ---
    if all_metric_computers_for_analysis: # Check if there is data to process
        base_results_dir_for_cross_analysis = os.path.join(os.getcwd(), "_results")
        cross_sim_processor = CrossSimProcessor(all_metric_computers_for_analysis, base_results_dir_for_cross_analysis)
        cross_sim_processor.plot_model_performance_vs_inputs()
        cross_sim_processor.plot_execution_time_vs_inputs_by_topology()
        cross_sim_processor.plot_execution_time_vs_inputs_by_chiplet_mapping()
        cross_sim_processor.plot_avg_model_latency_vs_inputs_by_topology()
        cross_sim_processor.plot_execution_time_percent_difference_by_topology()
        cross_sim_processor.plot_utilization_vs_inputs()
        cross_sim_processor.plot_latency_vs_utilization()
    else:
        print("\n" + "="*80)
        print("⚠️ No data was collected from any simulation run. Skipping cross-simulation analysis.")
        print("="*80)
    
    overall_duration = time.time() - overall_start_time
    print(f"Total script runtime: {overall_duration:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 