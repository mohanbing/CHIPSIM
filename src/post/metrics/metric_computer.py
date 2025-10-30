#!/usr/bin/env python3
# class_MetricComputer.py

import json
import os

import numpy as np

class MetricComputer:
    """
    Computes performance metrics based on simulation results.
    
    Args:
        retired_mapped_models (dict): Dictionary of completed mapped models
        global_time_us (float): Final global time of the simulation in microseconds
    """
    def __init__(self, 
                 retired_mapped_models, 
                 global_time_us,
                 num_chiplets,
                 dsent_stats_file_path=None):
        
        self.retired_mapped_models = retired_mapped_models
        self.global_time_us = global_time_us
        self.total_simulation_time_us = global_time_us  # Alias for cross-simulation compatibility
        self.num_chiplets = num_chiplets
        self.dsent_stats_file_path = dsent_stats_file_path
        self.dsent_stats = None
        
        # Extract all chiplets used across all models
        self.all_chiplets = set()
        for model in retired_mapped_models.values():
            self.all_chiplets.update(model.get_chiplets_used())

        # Initialize attributes to store computed metrics

        # ---- Utilization Metrics ----
        # Method 1: Total compute time based utilization
        self.per_chiplet_compute_time_utilization_pct = None  # Renamed from chiplet_utilization
        self.chiplet_total_compute_time_us = None  # Renamed from chiplet_compute_time
        self.mean_compute_time_chiplet_utilization_pct = None # Was compute_time_avg_utilization_pct

        # New attributes for activation communication time based utilization (Method 1 style)
        self.per_chiplet_activation_comm_time_utilization_pct = None
        self.chiplet_total_activation_comm_time_us = None
        self.mean_activation_comm_time_chiplet_utilization_pct = None

        # New attributes for weight loading communication time based utilization (Method 1 style)
        self.per_chiplet_weight_loading_time_utilization_pct = None
        self.chiplet_total_weight_loading_time_us = None
        self.mean_weight_loading_time_chiplet_utilization_pct = None

        # New attributes for combined (compute + activation_comm + weight_loading) time based utilization (Method 1 style)
        self.per_chiplet_combined_time_utilization_pct = None
        self.chiplet_total_combined_busy_time_us = None # Sum of (compute_latency + activation_comm_latency + weight_loading_latency) for tasks
        self.mean_combined_time_chiplet_utilization_pct = None
        
        # Method 2: Time-step based activity tracking
        # Compute Utilization
        self.utilization_time_points = None # Shared across compute, communication, and combined
        self.chiplet_compute_activity_at_time_step = None # {time_point: {chiplet_id: activity (0 or 1), ...}}
        self.chiplet_compute_activity_over_time = None
        self.system_compute_utilization_over_time_pct = None
        
        # Activation Communication Utilization
        self.chiplet_activation_comm_activity_at_time_step = None
        self.chiplet_activation_comm_activity_over_time = None
        self.system_activation_comm_utilization_over_time_pct = None
        
        # Weight Loading Communication Utilization
        self.chiplet_weight_loading_activity_at_time_step = None
        self.chiplet_weight_loading_activity_over_time = None
        self.system_weight_loading_utilization_over_time_pct = None
        
        # Combined Utilization (Compute + Activation Communication + Weight Loading)
        self.chiplet_combined_activity_at_time_step = None
        self.chiplet_combined_activity_over_time = None
        self.system_combined_utilization_over_time_pct = None
        
        # Original detailed activity tracking (renamed) - this was `self.chiplet_activity_at_time_step` before
        # This specific attribute might be redundant if the new ones cover its purpose,
        # but I'm keeping its structure as per the original code's intent for "Method 3".
        # It was described as: self.chiplet_activity_at_time_step = None # {time_point: {chiplet_id: activity (0 or 1), ...}}
        # Let's rename it to avoid collision and clarify its original purpose if it's still needed.
        self.detailed_general_activity_at_time_step = None 
        
        self.model_summary_metrics = None
        
        # ---- Power Metrics ----
        self.power_time_points = None
        self.power_time_step_us = None
        self.chiplet_compute_power_over_time = None
        self.chiplet_activation_comm_power_over_time = None
        self.chiplet_weight_loading_power_over_time = None
        self.chiplet_total_power_over_time = None
        self.system_compute_power_over_time = None
        self.system_activation_comm_power_over_time = None
        self.system_weight_loading_power_over_time = None
        self.system_total_power_over_time = None
        
        # ---- Energy Metrics ----
        self.chiplet_compute_energy_uj = None
        self.chiplet_activation_comm_energy_uj = None
        self.chiplet_weight_loading_energy_uj = None
        self.chiplet_total_energy_uj = None
        self.system_compute_energy_uj = None
        self.system_activation_comm_energy_uj = None
        self.system_weight_loading_energy_uj = None
        self.system_total_energy_uj = None
        
        self.approach_comparison_metrics = {
            "avg_time_per_input_per_model_type_main_sim": None,
            "avg_time_per_input_per_model_type_comparison": None
        }

    def compute_avg_system_utilization(self):
        """
        Computes the system compute utilization per chiplet based on total compute time.
        
        This method calculates utilization as the percentage of time each chiplet
        spends on compute, communication, or combined (compute + communication) tasks
        over the total simulation time. It achieves this by summing all relevant latencies
        (compute, communication, or their sum for combined) for each chiplet across all
        models and dividing by the total simulation time.

        It updates the following instance attributes:
        - For Compute Time:
            - `self.chiplet_total_compute_time_us`
            - `self.per_chiplet_compute_time_utilization_pct`
            - `self.mean_compute_time_chiplet_utilization_pct`
        - For Communication Time:
            - `self.chiplet_total_communication_time_us`
            - `self.per_chiplet_communication_time_utilization_pct`
            - `self.mean_communication_time_chiplet_utilization_pct`
        - For Combined Time (sum of compute and communication task durations):
            - `self.chiplet_total_combined_busy_time_us`
            - `self.per_chiplet_combined_time_utilization_pct`
            - `self.mean_combined_time_chiplet_utilization_pct`

        Note: This method sums latencies and does not account for overlapping execution
        intervals in the same way the `compute_all_utilization_over_time` method does.
        The assertion regarding 'layer_start_time_us' and 'layer_completion_time_us'
        is primarily for ensuring that models have detailed timing info if expected,
        but these specific start/completion times are not directly used in the latency summation here.

        Returns:
            None. Results are stored in instance attributes.
        """

        # Initialize dictionaries to store total times for each chiplet
        chiplet_total_compute_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_activation_comm_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_weight_loading_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}
        chiplet_total_combined_busy_time_us = {chiplet_id: 0.0 for chiplet_id in self.all_chiplets}

        # Iterate through each model
        for model_idx, mapped_model in self.retired_mapped_models.items():
            # Check if the model has per-input, per-layer start/completion times
            # (these are available when parallelization is enabled)
            # Also check for latency information.
            required_attrs = ['layer_start_time_us', 'layer_completion_time_us', 
                              'layer_compute_latency', 'layer_activation_comm_latency', 'layer_weight_loading_latency']
            for attr in required_attrs:
                assert hasattr(mapped_model, attr), \
                       f"Model {model_idx} ({mapped_model.model_name if hasattr(mapped_model, 'model_name') else 'Unknown'}) is missing '{attr}' attribute. This is required for utilization calculation."

            # Get mapping details
            layer_to_chiplets = {}
            for layer_idx, mappings in mapped_model.mapping:
                layer_to_chiplets[layer_idx] = mappings

            # Use detailed layer timing for more accurate utilization calculation
            for input_idx in range(mapped_model.num_inputs):
                for layer_idx in range(mapped_model.num_layers):
                    compute_latency = mapped_model.layer_compute_latency[input_idx].get(layer_idx, 0.0)
                    activation_comm_latency = mapped_model.layer_activation_comm_latency[input_idx].get(layer_idx, 0.0)
                    weight_loading_latency = mapped_model.layer_weight_loading_latency[input_idx].get(layer_idx, 0.0)

                    # Validate latencies
                    if not isinstance(compute_latency, (int, float)): compute_latency = 0.0
                    if not isinstance(activation_comm_latency, (int, float)): activation_comm_latency = 0.0
                    if not isinstance(weight_loading_latency, (int, float)): weight_loading_latency = 0.0
                    compute_latency = max(0.0, compute_latency)
                    activation_comm_latency = max(0.0, activation_comm_latency)
                    weight_loading_latency = max(0.0, weight_loading_latency)

                    # Process if there's any activity for this layer
                    if not (compute_latency > 0 or activation_comm_latency > 0 or weight_loading_latency > 0):
                        continue # Skip if all are zero

                    # Check if this layer has chiplet mappings
                    if layer_idx in layer_to_chiplets:
                        # Add time contribution for each chiplet involved
                        for chiplet_id, percentage in layer_to_chiplets[layer_idx]:
                            if chiplet_id not in self.all_chiplets: # Should not happen if all_chiplets is comprehensive
                                print(f"Warning: Chiplet ID {chiplet_id} from mapping not in self.all_chiplets. Skipping.")
                                continue
                            
                            if compute_latency > 0:
                                chiplet_total_compute_time_us[chiplet_id] += compute_latency
                            if activation_comm_latency > 0:
                                chiplet_total_activation_comm_time_us[chiplet_id] += activation_comm_latency
                            if weight_loading_latency > 0:
                                chiplet_total_weight_loading_time_us[chiplet_id] += weight_loading_latency
                            # Combined busy time adds all three, as they are sequential components of work for this task segment
                            chiplet_total_combined_busy_time_us[chiplet_id] += (compute_latency + activation_comm_latency + weight_loading_latency)

        # --- Store total times ---
        self.chiplet_total_compute_time_us = chiplet_total_compute_time_us
        self.chiplet_total_activation_comm_time_us = chiplet_total_activation_comm_time_us
        self.chiplet_total_weight_loading_time_us = chiplet_total_weight_loading_time_us
        self.chiplet_total_combined_busy_time_us = chiplet_total_combined_busy_time_us

        # --- Calculate and store per-chiplet utilization percentages ---
        per_chiplet_compute_utilization_pct = {}
        for chiplet_id, compute_time in chiplet_total_compute_time_us.items():
            util = (compute_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_compute_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_compute_time_utilization_pct = per_chiplet_compute_utilization_pct

        per_chiplet_activation_comm_utilization_pct = {}
        for chiplet_id, activation_comm_time in chiplet_total_activation_comm_time_us.items():
            util = (activation_comm_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_activation_comm_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_activation_comm_time_utilization_pct = per_chiplet_activation_comm_utilization_pct

        per_chiplet_weight_loading_utilization_pct = {}
        for chiplet_id, weight_loading_time in chiplet_total_weight_loading_time_us.items():
            util = (weight_loading_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_weight_loading_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_weight_loading_time_utilization_pct = per_chiplet_weight_loading_utilization_pct

        per_chiplet_combined_utilization_pct = {}
        for chiplet_id, combined_time in chiplet_total_combined_busy_time_us.items():
            util = (combined_time / self.global_time_us) * 100.0 if self.global_time_us > 0 else 0.0
            per_chiplet_combined_utilization_pct[chiplet_id] = min(100.0, util)
        self.per_chiplet_combined_time_utilization_pct = per_chiplet_combined_utilization_pct

        # --- Assertions for accumulated times ---
        if self.retired_mapped_models:
            total_accumulated_compute = sum(chiplet_total_compute_time_us.values())
            total_accumulated_activation_comm = sum(chiplet_total_activation_comm_time_us.values())
            total_accumulated_weight_loading = sum(chiplet_total_weight_loading_time_us.values())
            total_accumulated_combined = sum(chiplet_total_combined_busy_time_us.values())

            # Check if any activity was expected to be recorded
            # This assertion might be too strict if a sim genuinely has zero activity for a type.
            # For now, let's ensure that if there *was* activity, util isn't unexpectedly zero.
            if any(val > 0 for val in chiplet_total_compute_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_compute_utilization_pct.values()), \
                     "Non-zero total compute time but all per-chiplet compute utilizations are zero."
            if any(val > 0 for val in chiplet_total_activation_comm_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_activation_comm_utilization_pct.values()), \
                     "Non-zero total activation communication time but all per-chiplet activation communication utilizations are zero."
            if any(val > 0 for val in chiplet_total_weight_loading_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_weight_loading_utilization_pct.values()), \
                     "Non-zero total weight loading time but all per-chiplet weight loading utilizations are zero."
            if any(val > 0 for val in chiplet_total_combined_busy_time_us.values()):
                 assert any(val > 0 for val in per_chiplet_combined_utilization_pct.values()), \
                     "Non-zero total combined time but all per-chiplet combined utilizations are zero."

        # --- Calculate and store mean of per-chiplet utilizations ---
        if per_chiplet_compute_utilization_pct:
            self.mean_compute_time_chiplet_utilization_pct = sum(per_chiplet_compute_utilization_pct.values()) / len(per_chiplet_compute_utilization_pct)
        else:
            self.mean_compute_time_chiplet_utilization_pct = 0.0 # Or -1.0, consistent with previous behavior

        if per_chiplet_activation_comm_utilization_pct:
            self.mean_activation_comm_time_chiplet_utilization_pct = sum(per_chiplet_activation_comm_utilization_pct.values()) / len(per_chiplet_activation_comm_utilization_pct)
        else:
            self.mean_activation_comm_time_chiplet_utilization_pct = 0.0

        if per_chiplet_weight_loading_utilization_pct:
            self.mean_weight_loading_time_chiplet_utilization_pct = sum(per_chiplet_weight_loading_utilization_pct.values()) / len(per_chiplet_weight_loading_utilization_pct)
        else:
            self.mean_weight_loading_time_chiplet_utilization_pct = 0.0

        if per_chiplet_combined_utilization_pct:
            self.mean_combined_time_chiplet_utilization_pct = sum(per_chiplet_combined_utilization_pct.values()) / len(per_chiplet_combined_utilization_pct)
        else:
            self.mean_combined_time_chiplet_utilization_pct = 0.0

        # Add combined communication utilization for cross-simulation compatibility
        # This combines activation communication and weight loading utilization
        if per_chiplet_activation_comm_utilization_pct and per_chiplet_weight_loading_utilization_pct:
            combined_comm_utilization = {}
            for chiplet_id in per_chiplet_activation_comm_utilization_pct:
                activation_comm_util = per_chiplet_activation_comm_utilization_pct.get(chiplet_id, 0.0)
                weight_loading_util = per_chiplet_weight_loading_utilization_pct.get(chiplet_id, 0.0)
                combined_comm_utilization[chiplet_id] = activation_comm_util + weight_loading_util
            self.mean_communication_time_chiplet_utilization_pct = sum(combined_comm_utilization.values()) / len(combined_comm_utilization)
        else:
            self.mean_communication_time_chiplet_utilization_pct = 0.0

        return # Method updates instance attributes, returns None

    def compute_utilization_over_time(self, time_step_us=1.0):
        """
        Computes chiplet COMPUTE, COMMUNICATION, and COMBINED activity at each discrete time step
        and derives utilization metrics, storing the results in instance attributes.

        This method calculates:
        1.  Activity of each chiplet at each time step for compute, communication, and combined states.
        2.  Aggregated activity over time for each chiplet.
        3.  System-wide utilization percentage at each time point.
        4.  Average utilization percentage per chiplet over the simulation duration.
        5.  Overall average system utilization percentage across all time steps.

        Updates the following instance attributes:
            - `self.utilization_time_points`: Shared time points for all utilization calculations.
            - Compute Metrics:
                - `self.chiplet_compute_activity_at_time_step`
                - `self.chiplet_compute_activity_over_time`
                - `self.system_compute_utilization_over_time_pct`
            - Communication Metrics:
                - `self.chiplet_communication_activity_at_time_step`
                - `self.chiplet_communication_activity_over_time`
                - `self.system_communication_utilization_over_time_pct`
            - Combined Metrics:
                - `self.chiplet_combined_activity_at_time_step`
                - `self.chiplet_combined_activity_over_time`
                - `self.system_combined_utilization_over_time_pct`

        Args:
            time_step_us (float): Time step in microseconds for sampling utilization.

        Returns:
            None
        """
        
        # --- Helper to set all utilization attributes to default/empty ---
        def _set_default_utilization_metrics(time_points_val=np.array([]), activity_val={}, list_val=[], dict_val={}, float_val=0.0, num_tp=0, chiplet_ids_for_avg=None):
            if chiplet_ids_for_avg is None:
                chiplet_ids_for_avg = set()

            self.utilization_time_points = time_points_val
            
            self.chiplet_compute_activity_at_time_step = activity_val
            self.chiplet_compute_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_compute_utilization_over_time_pct = list_val
            
            self.chiplet_activation_comm_activity_at_time_step = activity_val
            self.chiplet_activation_comm_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_activation_comm_utilization_over_time_pct = list_val
            
            self.chiplet_weight_loading_activity_at_time_step = activity_val
            self.chiplet_weight_loading_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_weight_loading_utilization_over_time_pct = list_val
            
            self.chiplet_combined_activity_at_time_step = activity_val
            self.chiplet_combined_activity_over_time = {cid: [0]*num_tp for cid in chiplet_ids_for_avg} if num_tp > 0 else {}
            self.system_combined_utilization_over_time_pct = list_val

        # --- Handle Empty Input or Invalid Parameters ---
        if not self.retired_mapped_models or self.global_time_us <= 0 or time_step_us <= 0:
            _set_default_utilization_metrics()
            return

        # --- Build the set of all unique chiplet IDs involved ---
        all_system_chiplet_ids = set()
        total_system_chiplets_config = self.num_chiplets
        if total_system_chiplets_config <= 0:
            print(f"Warning: Total configured chiplets reported as {total_system_chiplets_config}. Check MetricComputer initialization. This may affect system utilization percentages.")
            total_system_chiplets_config = 0  # Treat as invalid for calculation if necessary

        for model in self.retired_mapped_models.values():
            if not hasattr(model, 'mapping'):
                print(f"Warning: Model {model.model_idx if hasattr(model, 'model_idx') else 'Unknown'} is missing 'mapping' attribute. Skipping for chiplet ID collection.")
                continue
            for layer_idx, mappings in model.mapping:
                for chiplet_id, percentage in mappings:
                    all_system_chiplet_ids.add(chiplet_id)

        # --- Determine effective total chiplets for utilization denominator ---
        if total_system_chiplets_config > 0:
            effective_total_chiplets_for_util = total_system_chiplets_config
        else:
            print("Warning: Using count of *used* chiplets for system utilization denominator due to invalid or zero configured total.")
            effective_total_chiplets_for_util = len(all_system_chiplet_ids)


        # --- Handle case where no chiplets are found or effective total is zero ---
        if not all_system_chiplet_ids:
            print("Warning: No chiplets found in any model mappings. All utilization metrics will be zero or empty.")
            # Time points are still generated, system utilization is 0%, per-chiplet metrics are empty.
            time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
            num_time_points = len(time_points)
            _set_default_utilization_metrics(
                time_points_val=time_points,
                activity_val={tp: {} for tp in time_points}, # Activity is empty per time point
                list_val=[0.0] * num_time_points if num_time_points > 0 else [],
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # No chiplets, so empty
                float_val=0.0,
                num_tp=num_time_points,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            return

        # --- Create time points ---
        time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
        self.utilization_time_points = time_points
        num_time_points = len(time_points)

        if num_time_points == 0:
            print("Warning: No time points generated (global_time_us might be too small or time_step_us invalid). All utilization metrics will be zero or empty.")
            _set_default_utilization_metrics(
                time_points_val=time_points, # empty np.array
                activity_val={},
                list_val=[],
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # Avg util is 0 if no time
                float_val=0.0,
                num_tp=0,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            return

        if effective_total_chiplets_for_util == 0:
            print("Warning: effective_total_chiplets_for_util is zero. System utilization cannot be computed accurately and will be reported as 0%.")
            # Chiplet activity can still be tracked, but system-wide utilization is ill-defined or zero.
            _set_default_utilization_metrics(
                time_points_val=time_points,
                activity_val={tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points},
                list_val=[0.0] * num_time_points, # System util is 0
                dict_val={cid: 0.0 for cid in all_system_chiplet_ids}, # Chiplet avg can still be calculated based on its own activity
                float_val=0.0, # System avg is 0
                num_tp=num_time_points,
                chiplet_ids_for_avg=all_system_chiplet_ids
            )
            # Continue to calculate per-chiplet activity, but system metrics will be 0.
            # The _calculate_derived_utilization_metrics helper will handle division by zero for system util.

        # --- Initialize primary activity dictionaries ---
        chiplet_compute_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_activation_comm_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_weight_loading_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        chiplet_combined_activity_at_time_step = {tp: {cid: 0 for cid in all_system_chiplet_ids} for tp in time_points}
        
        epsilon = 1e-9 # For ceiling operations

        # --- Populate activity dictionaries by iterating through computations and communications ---
        for model_idx, mapped_model in self.retired_mapped_models.items():
            if not (hasattr(mapped_model, 'mapping') and
                    hasattr(mapped_model, 'num_inputs') and
                    hasattr(mapped_model, 'num_layers') and
                    hasattr(mapped_model, 'layer_start_time_us') and
                    hasattr(mapped_model, 'layer_compute_latency') and # Needed for compute and combined
                    hasattr(mapped_model, 'layer_activation_comm_latency') and # Needed for activation communication and combined
                    hasattr(mapped_model, 'layer_weight_loading_latency')): # Needed for weight loading and combined
                print(f"Warning: Model {model_idx} ({mapped_model.model_name if hasattr(mapped_model, 'model_name') else 'Unknown'}) is missing required attributes for detailed utilization calculation. Skipping this model.")
                continue

            layer_to_chiplets_map = {layer_idx: mappings for layer_idx, mappings in mapped_model.mapping}

            for input_idx in range(mapped_model.num_inputs):
                # Check if timing dictionaries exist for this input_idx
                if not (input_idx in mapped_model.layer_start_time_us and
                        input_idx in mapped_model.layer_compute_latency and
                        input_idx in mapped_model.layer_activation_comm_latency and
                        input_idx in mapped_model.layer_weight_loading_latency):
                    # This might happen if an input didn't run fully or data is missing
                    continue

                for layer_idx in range(mapped_model.num_layers):
                    layer_start_time = mapped_model.layer_start_time_us[input_idx].get(layer_idx, -1)
                    compute_latency = mapped_model.layer_compute_latency[input_idx].get(layer_idx, 0)
                    activation_comm_latency = mapped_model.layer_activation_comm_latency[input_idx].get(layer_idx, 0)
                    weight_loading_latency = mapped_model.layer_weight_loading_latency[input_idx].get(layer_idx, 0)

                    # Basic validation of timing data
                    if not (isinstance(layer_start_time, (int, float)) and
                            isinstance(compute_latency, (int, float)) and
                            isinstance(activation_comm_latency, (int, float)) and
                            isinstance(weight_loading_latency, (int, float))):
                        # print(f"Warning: Invalid timing data types for Model {model_idx}, Input {input_idx}, Layer {layer_idx}. Skipping.")
                        continue
                    
                    if compute_latency < 0: compute_latency = 0 # Treat negative as zero
                    if activation_comm_latency < 0: activation_comm_latency = 0 # Treat negative as zero
                    if weight_loading_latency < 0: weight_loading_latency = 0 # Treat negative as zero

                    chiplets_for_layer = {cid for cid, _ in layer_to_chiplets_map.get(layer_idx, [])}
                    if not chiplets_for_layer:
                        continue

                    # --- Compute Activity ---
                    if layer_start_time >= 0 and compute_latency > 0:
                        layer_compute_end_time = layer_start_time + compute_latency
                        start_tp_idx = max(0, int(np.floor(layer_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((layer_compute_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_compute_activity_at_time_step[tp]:
                                    chiplet_compute_activity_at_time_step[tp][cid] = 1
                    
                    # --- Activation Communication Activity ---
                    if layer_start_time >= 0 and activation_comm_latency > 0: # compute_latency can be 0 if only communication happens after a data arrival
                        activation_comm_start_time = layer_start_time + compute_latency # Activation comm starts after compute finishes
                        activation_comm_end_time = activation_comm_start_time + activation_comm_latency
                        start_tp_idx = max(0, int(np.floor(activation_comm_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((activation_comm_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_activation_comm_activity_at_time_step[tp]:
                                    chiplet_activation_comm_activity_at_time_step[tp][cid] = 1

                    # --- Weight Loading Communication Activity ---
                    if layer_start_time >= 0 and weight_loading_latency > 0:
                        # Weight loading can occur at any time, not necessarily after compute
                        # For now, assume it occurs before compute (typical weight loading pattern)
                        weight_loading_start_time = layer_start_time - weight_loading_latency
                        weight_loading_end_time = layer_start_time
                        start_tp_idx = max(0, int(np.floor(weight_loading_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((weight_loading_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_weight_loading_activity_at_time_step[tp]:
                                    chiplet_weight_loading_activity_at_time_step[tp][cid] = 1

                    # --- Combined Activity (Compute OR Activation Communication OR Weight Loading) ---
                    # A chiplet is active if it's computing OR doing activation communication OR weight loading.
                    # The interval spans from weight loading start to end of activation communication.
                    if layer_start_time >= 0 and (compute_latency > 0 or activation_comm_latency > 0 or weight_loading_latency > 0):
                        # Combined activity spans from weight loading start to activation communication end
                        activity_start_time = layer_start_time - weight_loading_latency  # Start with weight loading
                        activity_end_time = layer_start_time + compute_latency + activation_comm_latency  # End with activation communication
                        
                        start_tp_idx = max(0, int(np.floor(activity_start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil((activity_end_time - epsilon) / time_step_us)))
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            tp = time_points[t_idx]
                            for cid in chiplets_for_layer:
                                if cid in chiplet_combined_activity_at_time_step[tp]:
                                    chiplet_combined_activity_at_time_step[tp][cid] = 1
        
        # --- Calculate Derived Metrics ---
        (self.chiplet_compute_activity_over_time,
         self.system_compute_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_compute_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_compute_activity_at_time_step = chiplet_compute_activity_at_time_step

        (self.chiplet_activation_comm_activity_over_time,
         self.system_activation_comm_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_activation_comm_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_activation_comm_activity_at_time_step = chiplet_activation_comm_activity_at_time_step

        (self.chiplet_weight_loading_activity_over_time,
         self.system_weight_loading_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_weight_loading_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_weight_loading_activity_at_time_step = chiplet_weight_loading_activity_at_time_step

        (self.chiplet_combined_activity_over_time,
         self.system_combined_utilization_over_time_pct) = self._calculate_derived_utilization_metrics(
            chiplet_combined_activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points
        )
        self.chiplet_combined_activity_at_time_step = chiplet_combined_activity_at_time_step
        
        return # Explicitly return None

    def _calculate_derived_utilization_metrics(self, activity_at_time_step, time_points, all_system_chiplet_ids, effective_total_chiplets_for_util, num_time_points):
        """
        Helper to calculate derived utilization metrics from activity data.
        """
        if num_time_points == 0: # Should be caught earlier, but as a safeguard
            activity_over_time = {cid: [] for cid in all_system_chiplet_ids}
            system_util_over_time_pct = []
            return activity_over_time, system_util_over_time_pct

        activity_over_time = {cid: [0] * num_time_points for cid in all_system_chiplet_ids}
        system_util_over_time_pct = [0.0] * num_time_points

        for t_idx, tp in enumerate(time_points):
            active_chiplets_count_at_tp = 0
            activity_dict_at_tp = activity_at_time_step.get(tp, {}) # Safe get
            
            for cid in all_system_chiplet_ids:
                activity = activity_dict_at_tp.get(cid, 0) # Safe get
                activity_over_time[cid][t_idx] = activity
                active_chiplets_count_at_tp += activity
            
            if effective_total_chiplets_for_util > 0:
                system_util_over_time_pct[t_idx] = (active_chiplets_count_at_tp / effective_total_chiplets_for_util) * 100.0
            else: # Denominator is zero, system utilization is undefined or 0%
                system_util_over_time_pct[t_idx] = 0.0

        return activity_over_time, system_util_over_time_pct

    def compute_model_summary_metrics(self):
        """
        Computes summary metrics for each model (latency, energy, traffic).
        Stores results in the `self.model_summary_metrics` attribute.
        Handles both pipelined and non-pipelined scenarios.
        
        Returns:
            dict: A dictionary containing summary metrics for each model.
                  Key: model_idx
                  Value: dict of metrics
        """
        
        # Assert that retired_mapped_models exists and is not empty
        assert self.retired_mapped_models, "ERROR: No retired models found. Metrics cannot be computed."
        
        self.model_summary_metrics = {}
        
        for model_idx, mapped_model in self.retired_mapped_models.items():
            metrics = {}
            
            # --- Basic attribute validation with detailed assertions ---
            model_name = getattr(mapped_model, 'model_name', f"Model {model_idx}")
            
            # Required attributes for metric computation
            assert hasattr(mapped_model, 'num_inputs'), f"ERROR: Model {model_idx} ({model_name}) missing 'num_inputs' attribute"
            assert mapped_model.num_inputs > 0, f"ERROR: Model {model_idx} ({model_name}) has invalid num_inputs: {mapped_model.num_inputs}"

            assert hasattr(mapped_model, 'layer_compute_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_compute_latency' attribute"
            assert hasattr(mapped_model, 'layer_activation_comm_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_activation_comm_latency' attribute"
            assert hasattr(mapped_model, 'layer_weight_loading_latency'), f"ERROR: Model {model_idx} ({model_name}) missing 'layer_weight_loading_latency' attribute"
            
            # Verify data integrity - at least one input should have latency data
            valid_compute_data = False
            valid_activation_comm_data = False
            valid_weight_loading_data = False
            
            for input_idx in range(mapped_model.num_inputs):
                if input_idx in mapped_model.layer_compute_latency:
                    valid_compute_data = True
                if input_idx in mapped_model.layer_activation_comm_latency:
                    valid_activation_comm_data = True
                if input_idx in mapped_model.layer_weight_loading_latency:
                    valid_weight_loading_data = True
            
            assert valid_compute_data, f"ERROR: Model {model_idx} ({model_name}) has no valid compute latency data for any input"
            assert valid_activation_comm_data, f"ERROR: Model {model_idx} ({model_name}) has no valid activation communication latency data for any input"
            assert valid_weight_loading_data, f"ERROR: Model {model_idx} ({model_name}) has no valid weight loading latency data for any input"

            # Initialize metrics
            max_single_input_total_latency = 0.0
            max_single_input_compute_latency = 0.0
            max_single_input_activation_comm_latency = 0.0
            max_single_input_weight_loading_latency = 0.0

            sum_compute_latency_all_inputs = 0.0
            sum_activation_comm_latency_all_inputs = 0.0
            sum_weight_loading_latency_all_inputs = 0.0
            total_compute_energy_all_inputs = 0.0
            total_activation_comm_traffic = 0
            total_weight_loading_traffic = 0

            num_valid_inputs = 0
            all_inputs_latency_data = []

            # Calculate totals and maximums across all inputs
            for input_idx in range(mapped_model.num_inputs):
                input_compute_latency = 0.0
                input_activation_comm_latency = 0.0
                input_weight_loading_latency = 0.0
                
                if input_idx in mapped_model.layer_compute_latency:
                    for layer_idx, latency in mapped_model.layer_compute_latency[input_idx].items():
                        if latency > 0:
                            input_compute_latency += latency
                
                if input_idx in mapped_model.layer_activation_comm_latency:
                    for layer_idx, latency in mapped_model.layer_activation_comm_latency[input_idx].items():
                        if latency > 0:
                            input_activation_comm_latency += latency
                
                if input_idx in mapped_model.layer_weight_loading_latency:
                    for layer_idx, latency in mapped_model.layer_weight_loading_latency[input_idx].items():
                        if latency > 0:
                            input_weight_loading_latency += latency
                
                # Only count this input if it has any positive latency values
                if input_compute_latency > 0 or input_activation_comm_latency > 0 or input_weight_loading_latency > 0:
                    num_valid_inputs += 1
                    input_total_latency = input_compute_latency + input_activation_comm_latency + input_weight_loading_latency
                    all_inputs_latency_data.append((input_idx, input_compute_latency, input_activation_comm_latency, input_weight_loading_latency, input_total_latency))
                    
                    # Track maximum single input latencies
                    max_single_input_total_latency = max(max_single_input_total_latency, input_total_latency)
                    max_single_input_compute_latency = max(max_single_input_compute_latency, input_compute_latency)
                    max_single_input_activation_comm_latency = max(max_single_input_activation_comm_latency, input_activation_comm_latency)
                    max_single_input_weight_loading_latency = max(max_single_input_weight_loading_latency, input_weight_loading_latency)
                    
                    # Accumulate sums across all inputs
                    sum_compute_latency_all_inputs += input_compute_latency
                    sum_activation_comm_latency_all_inputs += input_activation_comm_latency
                    sum_weight_loading_latency_all_inputs += input_weight_loading_latency
                
                # Accumulate energy (separate from latency check)
                if hasattr(mapped_model, 'layer_compute_energy') and input_idx in mapped_model.layer_compute_energy:
                    total_compute_energy_all_inputs += sum(energy for energy in mapped_model.layer_compute_energy[input_idx].values() if energy >= 0)

            # Assert that at least one input has valid latency data
            assert num_valid_inputs > 0, f"ERROR: Model {model_idx} ({model_name}) has no inputs with valid latency data"

            # Calculate total model latency (summed work)
            total_model_latency_summed_work = sum_compute_latency_all_inputs + sum_activation_comm_latency_all_inputs + sum_weight_loading_latency_all_inputs
            assert total_model_latency_summed_work > 0, f"ERROR: Model {model_idx} ({model_name}) has zero total latency"

            # Calculate total activation communication traffic safely
            if hasattr(mapped_model, 'layer_activation_comm_traffic'):
                for layer_idx, traffic_data in mapped_model.layer_activation_comm_traffic.items():
                    if isinstance(traffic_data, dict):
                        for src_chiplet, dest_map in traffic_data.items():
                            if isinstance(dest_map, dict):
                                total_activation_comm_traffic += sum(v for v in dest_map.values() if isinstance(v, (int, float)) and v >= 0)
                    elif isinstance(traffic_data, (int, float)) and traffic_data >= 0:
                        total_activation_comm_traffic += traffic_data

            # Calculate total weight loading traffic safely
            if hasattr(mapped_model, 'layer_weight_loading_traffic'):
                for layer_idx, traffic_data in mapped_model.layer_weight_loading_traffic.items():
                    if isinstance(traffic_data, dict):
                        for src_chiplet, dest_map in traffic_data.items():
                            if isinstance(dest_map, dict):
                                total_weight_loading_traffic += sum(v for v in dest_map.values() if isinstance(v, (int, float)) and v >= 0)
                    elif isinstance(traffic_data, (int, float)) and traffic_data >= 0:
                        total_weight_loading_traffic += traffic_data

            # Get overall model start/end times
            model_start_time = getattr(mapped_model, 'model_start_time_us', -1.0)
            model_completion_time = getattr(mapped_model, 'model_completion_time_us', -1.0)
            
            # Always ensure a valid duration is calculated
            if model_start_time >= 0 and model_completion_time >= model_start_time:
                overall_model_duration = model_completion_time - model_start_time
            elif hasattr(mapped_model, 'input_completion_time_us') and model_start_time >= 0:
                # Try to use input completion times as a fallback
                valid_completion_times = [t for t in mapped_model.input_completion_time_us if t >= 0]
                if valid_completion_times:
                    latest_completion = max(valid_completion_times)
                    overall_model_duration = latest_completion - model_start_time
                    print(f"INFO: Model {model_idx} ({model_name}): Using max input completion time for duration: {overall_model_duration:.2f} μs")
                else:
                    overall_model_duration = -1.0
            else:
                overall_model_duration = -1.0
                
            # At this point, metrics should be valid because of our assertions
            metrics['max_single_input_total_latency'] = max_single_input_total_latency
            metrics['max_single_input_compute_latency'] = max_single_input_compute_latency
            metrics['max_single_input_activation_comm_latency'] = max_single_input_activation_comm_latency
            metrics['max_single_input_weight_loading_latency'] = max_single_input_weight_loading_latency
            metrics['sum_compute_latency_all_inputs'] = sum_compute_latency_all_inputs
            metrics['sum_activation_comm_latency_all_inputs'] = sum_activation_comm_latency_all_inputs
            metrics['sum_weight_loading_latency_all_inputs'] = sum_weight_loading_latency_all_inputs
            metrics['total_compute_energy_all_inputs'] = total_compute_energy_all_inputs
            metrics['total_model_latency_summed_work'] = total_model_latency_summed_work
            metrics['total_activation_comm_traffic'] = total_activation_comm_traffic
            metrics['total_weight_loading_traffic'] = total_weight_loading_traffic
            metrics['overall_model_duration'] = overall_model_duration
            
            # Store the metrics in our dictionary
            self.model_summary_metrics[model_idx] = metrics
            print(f"Successfully computed metrics for Model {model_idx} ({model_name})")

        # Verify metrics were computed for all models
        assert len(self.model_summary_metrics) == len(self.retired_mapped_models), \
               f"ERROR: Metrics computed for only {len(self.model_summary_metrics)} models out of {len(self.retired_mapped_models)}"

        print(f"Computed metrics for all {len(self.model_summary_metrics)} models")
        return self.model_summary_metrics

    def compute_approach_comparison_metrics(self, individual_results, empty_system_individual_model_results):
        """
        Computes model execution times and averages, including separate compute and communication metrics.
        NEW: Compares average execution time per input for each model type across three sources:
             1. Main simulation (self.retired_mapped_models)
             2. Individual model (main sim compute + communication from individual_results)
             3. Empty system model (from empty_system_individual_model_results)
        Also computes separate average compute and communication times per model type.
        Stores results in the `self.approach_comparison_metrics` attribute dictionary.
        
        Args:
            individual_results (Optional[dict]): Results from the individual layer communication simulation.
                                                 Expected format: {(model_idx, layer_idx): {"results": {\'total_runtime_us\': float}}}
                                                 If None, communication from this source is considered zero.
            empty_system_individual_model_results (Optional[dict]): Pre-calculated results for model types in an empty system.
                                                                    Expected format: {model_name: {"avg_time_per_input": float, 
                                                                                                    "avg_compute_time_per_input": float, 
                                                                                                    "avg_comm_time_per_input": float, 
                                                                                                    "total_inputs": int}}
                                                                    If None, metrics from this source will be None.
        Returns:
            dict: The `self.approach_comparison_metrics` dictionary containing "model_type_performance_comparison".
                  This key holds a breakdown for each model type, including:
                  - Main simulation metrics (avg total/compute/communication time per input, totals, instances)
                  - Individual model metrics (avg total time per input)
                  - Empty system model metrics (avg total/compute/communication time per input, total inputs)
        """
        
        # ====================================================
        # Initialize data structures for aggregating model metrics by type
        # ====================================================
        main_simulation_aggregated_data = {}  # Aggregates data from main simulation by model type
        individual_model_aggregated_data = {}  # Aggregates individual model data by model type

        # ====================================================
        # Process each model instance to extract timing data
        # ====================================================
        for model_idx, mapped_model in self.retired_mapped_models.items():
            # Initialize timing accumulators for this model instance
            model_instance_compute_time = 0.0
            model_instance_comm_time = 0.0

            # Get model type name for grouping
            model_type_name = mapped_model.model_name if hasattr(mapped_model, 'model_name') else f"Model {model_idx}"

            # Initialize aggregated data structures for new model types
            if model_type_name not in main_simulation_aggregated_data:
                main_simulation_aggregated_data[model_type_name] = {
                    "total_time": 0.0, 
                    "total_compute_time": 0.0, 
                    "total_comm_time": 0.0, 
                    "total_inputs": 0, 
                    "model_instances": []
                }
            
            if model_type_name not in individual_model_aggregated_data:
                individual_model_aggregated_data[model_type_name] = {
                    "total_simulated_time": 0.0, 
                    "total_inputs_processed": 0
                }

            # ====================================================
            # Extract compute latencies from main simulation
            # ====================================================
            processed_inputs = set() 
            if hasattr(mapped_model, 'layer_compute_latency') and mapped_model.layer_compute_latency:
                for input_idx, layer_compute_latencies in mapped_model.layer_compute_latency.items():
                    processed_inputs.add(input_idx)
                    for compute_latency in layer_compute_latencies.values():
                        if compute_latency > 0:
                            model_instance_compute_time += compute_latency
            
            # ====================================================
            # Extract activation communication latencies from main simulation
            # ====================================================
            if hasattr(mapped_model, 'layer_activation_comm_latency') and mapped_model.layer_activation_comm_latency:
                for input_idx, layer_activation_comm_latencies in mapped_model.layer_activation_comm_latency.items():
                    processed_inputs.add(input_idx)
                    for layer_idx, main_sim_activation_comm_latency in layer_activation_comm_latencies.items():
                        if main_sim_activation_comm_latency >= 0:  # Skip invalid latencies (-1)
                            model_instance_comm_time += main_sim_activation_comm_latency

            # ====================================================
            # Extract weight loading latencies from main simulation
            # ====================================================
            if hasattr(mapped_model, 'layer_weight_loading_latency') and mapped_model.layer_weight_loading_latency:
                for input_idx, layer_weight_loading_latencies in mapped_model.layer_weight_loading_latency.items():
                    processed_inputs.add(input_idx)
                    for layer_idx, main_sim_weight_loading_latency in layer_weight_loading_latencies.items():
                        if main_sim_weight_loading_latency >= 0:  # Skip invalid latencies (-1)
                            model_instance_comm_time += main_sim_weight_loading_latency
            
            # ====================================================
            # Determine number of inputs processed by this model instance
            # ====================================================
            num_inputs_processed = len(processed_inputs) if processed_inputs else \
                                  (mapped_model.num_inputs if hasattr(mapped_model, 'num_inputs') and mapped_model.num_inputs > 0 else 1)
            if num_inputs_processed == 0: 
                num_inputs_processed = 1

            # ====================================================
            # Aggregate main simulation data by model type
            # ====================================================
            model_instance_total_time = model_instance_compute_time + model_instance_comm_time
            
            main_simulation_aggregated_data[model_type_name]["total_time"] += model_instance_total_time
            main_simulation_aggregated_data[model_type_name]["total_compute_time"] += model_instance_compute_time
            main_simulation_aggregated_data[model_type_name]["total_comm_time"] += model_instance_comm_time
            main_simulation_aggregated_data[model_type_name]["total_inputs"] += num_inputs_processed
            main_simulation_aggregated_data[model_type_name]["model_instances"].append(model_idx)

            # ====================================================
            # Extract individual model communication latencies (if available)
            # ====================================================
            individual_model_comm_time = 0.0
            if hasattr(mapped_model, 'layer_compute_latency') and mapped_model.layer_compute_latency:
                # Iterate through computed layers to sum up their individual communication latencies
                for input_idx, computed_layers in mapped_model.layer_compute_latency.items():
                    for layer_idx, compute_latency_value in computed_layers.items():
                        if compute_latency_value > 0:  # If layer was actually computed for this input
                            individual_results_key = (model_idx, layer_idx) 
                            if individual_results:
                                individual_layer_data = individual_results.get(individual_results_key)
                                if individual_layer_data and "results" in individual_layer_data:
                                    individual_comm_latency = individual_layer_data["results"].get('total_runtime_us')
                                    if individual_comm_latency is not None and individual_comm_latency >= 0:
                                        individual_model_comm_time += individual_comm_latency
            
            # ====================================================
            # Aggregate individual model data by model type
            # ====================================================
            individual_model_total_time = model_instance_compute_time + individual_model_comm_time
            
            individual_model_aggregated_data[model_type_name]["total_simulated_time"] += individual_model_total_time
            individual_model_aggregated_data[model_type_name]["total_inputs_processed"] += num_inputs_processed

        # ====================================================
        # Calculate average metrics for main simulation data
        # ====================================================
        main_simulation_averages_by_type = {}
        for model_type_name, aggregated_data in main_simulation_aggregated_data.items():
            total_inputs = aggregated_data["total_inputs"]
            if total_inputs > 0:
                avg_total_time = aggregated_data["total_time"] / total_inputs
                avg_compute_time = aggregated_data["total_compute_time"] / total_inputs
                avg_comm_time = aggregated_data["total_comm_time"] / total_inputs
            else:
                avg_total_time = avg_compute_time = avg_comm_time = 0.0
                
            main_simulation_averages_by_type[model_type_name] = {
                "avg_time_per_input": avg_total_time, 
                "avg_compute_time_per_input": avg_compute_time, 
                "avg_comm_time_per_input": avg_comm_time, 
                "total_inputs": total_inputs, 
                "total_time": aggregated_data["total_time"], 
                "total_compute_time": aggregated_data["total_compute_time"],
                "total_comm_time": aggregated_data["total_comm_time"], 
                "model_instances": aggregated_data["model_instances"]
            }

        # ====================================================
        # Calculate average metrics for individual model data
        # ====================================================
        individual_model_averages_by_type = {}
        # Calculate averages from individual_model_aggregated_data.
        # This data includes model_instance_compute_time + individual_model_comm_time,
        # where individual_model_comm_time is 0 if individual_results was None.
        for model_type_name, aggregated_data in individual_model_aggregated_data.items():
            total_inputs_processed = aggregated_data["total_inputs_processed"]
            if total_inputs_processed > 0:
                avg_execution_time = aggregated_data["total_simulated_time"] / total_inputs_processed
                individual_model_averages_by_type[model_type_name] = avg_execution_time
            else:
                individual_model_averages_by_type[model_type_name] = None 
        
        # ====================================================
        # Create comprehensive performance comparison across all approaches
        # ====================================================
        comprehensive_approach_comparison = {}
        all_model_type_names = set(main_simulation_averages_by_type.keys()) | \
                                 set(individual_model_averages_by_type.keys())
        if empty_system_individual_model_results:
            all_model_type_names.update(empty_system_individual_model_results.keys())

        for model_type_name in all_model_type_names:
            # Get main simulation data (with detailed breakdown)
            main_sim_data = main_simulation_averages_by_type.get(model_type_name, {})
            
            # Extract individual model average time
            if individual_results is None:
                individual_model_avg_time = 0.0  # Set to 0.0 if no individual results were provided
            else:
                # Otherwise, use the calculated average from individual_model_averages_by_type
                # This might be None if total_inputs_processed was 0 for this model_type_name.
                individual_model_avg_time = individual_model_averages_by_type.get(model_type_name)
            
            # Extract empty system model metrics
            empty_system_model_metrics = {
                "avg_time_per_input": None,
                "avg_compute_time_per_input": None,
                "avg_comm_time_per_input": None,
                "total_inputs": None  # This will remain None as it's not in the source structure
            }
            if empty_system_individual_model_results:
                # Get data for the specific model type from the empty system results
                model_data_for_empty_system = empty_system_individual_model_results.get(model_type_name, {})
                
                # Get the 'latency' sub-dictionary, defaulting to an empty dict if not found
                latency_values_for_empty_system = model_data_for_empty_system.get("latency", {})
                
                # Ensure latency_values_for_empty_system is a dictionary before trying to get items from it
                if isinstance(latency_values_for_empty_system, dict):
                    empty_system_model_metrics["avg_time_per_input"] = latency_values_for_empty_system.get("total")
                    empty_system_model_metrics["avg_compute_time_per_input"] = latency_values_for_empty_system.get("compute")
                    empty_system_model_metrics["avg_comm_time_per_input"] = latency_values_for_empty_system.get("communication")
                
                # 'total_inputs' is not available in the source data structure for empty system results,
                # so empty_system_model_metrics["total_inputs"] remains None as initialized.

            # Store comprehensive comparison data with detailed main simulation breakdown
            comprehensive_approach_comparison[model_type_name] = {
                # Main simulation detailed metrics (includes compute/communication breakdown)
                "main_simulation": {
                    "avg_time_per_input": main_sim_data.get("avg_time_per_input"),
                    "avg_compute_time_per_input": main_sim_data.get("avg_compute_time_per_input"),
                    "avg_comm_time_per_input": main_sim_data.get("avg_comm_time_per_input"),
                    "total_inputs": main_sim_data.get("total_inputs"),
                    "total_time": main_sim_data.get("total_time"),
                    "total_compute_time": main_sim_data.get("total_compute_time"),
                    "total_comm_time": main_sim_data.get("total_comm_time"),
                    "model_instances": main_sim_data.get("model_instances")
                },
                # Individual model metrics
                "individual_model": {
                    "avg_time_per_input": individual_model_avg_time
                },
                # Empty system model metrics
                "empty_system_model": empty_system_model_metrics
            }
            
        # ====================================================
        # Store results in instance attributes (single comprehensive metric)
        # ====================================================
        self.approach_comparison_metrics["model_type_performance_comparison"] = comprehensive_approach_comparison

        return self.approach_comparison_metrics

    def _load_dsent_stats(self):
        """
        Loads DSENT stats from the file specified in self.dsent_stats_file_path.
        This method is idempotent; it will only load the file once.
        """
        # If stats are already loaded, do nothing.
        if self.dsent_stats is not None:
            return

        # If no file path is provided, initialize as an empty list and return.
        if not self.dsent_stats_file_path:
            print("Warning: DSENT stats file path not provided. Power metrics will be based on zero communication power.")
            self.dsent_stats = []
            return

        # Check if the file exists.
        if not os.path.exists(self.dsent_stats_file_path):
            print(f"Warning: DSENT stats file not found at '{self.dsent_stats_file_path}'. Power metrics will be based on zero communication power.")
            self.dsent_stats = []
            return
            
        # Load the stats from the JSONL file.
        loaded_stats = []
        try:
            with open(self.dsent_stats_file_path, 'r') as f:
                for line in f:
                    try:
                        loaded_stats.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from a line in {self.dsent_stats_file_path}. Skipping line.")
            self.dsent_stats = loaded_stats
            print(f"Successfully loaded {len(self.dsent_stats)} entries from {self.dsent_stats_file_path}.")
        except Exception as e:
            print(f"Error: Failed to read DSENT stats file '{self.dsent_stats_file_path}': {e}")
            # Initialize as empty list on failure to prevent repeated attempts.
            self.dsent_stats = []

    def compute_power_profile(self, time_step_us=1.0):
        """
        Compute the power profile of the system over time, considering compute and communication.
        Uses chunk-level results for accurate per-chiplet power calculation.
        
        Args:
            time_step_us (float): The time step for the power profile in microseconds.
        """
        # Store the time step for later use in energy calculations
        self.power_time_step_us = time_step_us
        
        # Load DSENT stats if they haven't been loaded yet.
        self._load_dsent_stats()
        dsent_stats = self.dsent_stats

        # Determine the full set of chiplets in the system. All chiplets consume
        # communication power, even if idle.
        if self.num_chiplets > 0:
            all_system_chiplets = set(range(1, self.num_chiplets + 1))
        else:
            # Fallback to used chiplets if total number is not specified
            all_system_chiplets = self.all_chiplets
            if not all_system_chiplets:
                print("Warning: No chiplets to analyze (total configured number is zero and no chiplets were used). Power profile will be empty.")
            else:
                print(f"Warning: `num_chiplets` is not configured. Power metrics will only cover the {len(all_system_chiplets)} chiplets used in retired models.")

        if not all_system_chiplets:
            self.power_time_points = np.array([])
            self.chiplet_compute_power_over_time = {}
            self.chiplet_communication_power_over_time = {}
            self.chiplet_total_power_over_time = {}
            self.system_compute_power_over_time = np.array([])
            self.system_communication_power_over_time = np.array([])
            self.system_total_power_over_time = np.array([])
            return

        # --- Time points ---
        time_points = np.arange(0, self.global_time_us + time_step_us, time_step_us)
        num_time_points = len(time_points)
        epsilon = 1e-9

        # --- Initialize power profiles ---
        chiplet_compute_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}
        chiplet_activation_comm_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}
        chiplet_weight_loading_power_over_time = {cid: np.zeros(num_time_points) for cid in all_system_chiplets}

        # --- Process Compute Power using chunk results ---
        for model in self.retired_mapped_models.values():
            # Check if model has chunk_results
            if not hasattr(model, 'chunk_results'):
                continue
                
            for input_idx in range(model.num_inputs):
                if not all(hasattr(model, attr) for attr in ['layer_start_time_us']):
                    continue
                if not all(input_idx in getattr(model, attr) for attr in ['layer_start_time_us']):
                    continue

                for layer_idx in range(model.num_layers):
                    start_time = model.layer_start_time_us[input_idx].get(layer_idx)
                    
                    if start_time is None:
                        continue
                    
                    # Get chunk results for this layer
                    if layer_idx not in model.chunk_results:
                        continue
                    
                    # Process each chunk (chiplet) separately
                    for chiplet_id, (chunk_idx, chunk_result) in model.chunk_results[layer_idx].items():
                        # Get chunk-specific energy
                        chunk_energy_fj = chunk_result.get('energy_fj', 0)
                        if chunk_energy_fj <= 0:
                            continue
                        
                        # Get chunk-specific latency
                        chunk_latency_us = chunk_result.get('latency_us', 0)
                        if chunk_latency_us <= 0:
                            continue
                        
                        # Calculate chunk-specific power (convert fJ to J: 1e-15)
                        chunk_power_w = (chunk_energy_fj * 1e-15) / (chunk_latency_us * 1e-6)
                        
                        # The chunk is active from the layer start time for its own latency duration
                        activity_end_time = start_time + chunk_latency_us
                        
                        # Determine the range of time points this activity could affect
                        start_tp_idx = max(0, int(np.floor(start_time / time_step_us)))
                        end_tp_idx = min(num_time_points, int(np.ceil(activity_end_time / time_step_us)))
                        
                        # Add this chunk's power contribution to the chiplet's power profile
                        if chiplet_id in chiplet_compute_power_over_time:
                            for t_idx in range(start_tp_idx, end_tp_idx):
                                # Calculate the duration of overlap with this time bin
                                bin_start_time = time_points[t_idx]
                                bin_end_time = bin_start_time + time_step_us
                                
                                overlap_start = max(start_time, bin_start_time)
                                overlap_end = min(activity_end_time, bin_end_time)
                                overlap_duration = max(0, overlap_end - overlap_start)
                                
                                if overlap_duration > 0:
                                    # Prorate power: average power in this bin from this activity
                                    avg_power_in_bin = chunk_power_w * (overlap_duration / time_step_us)
                                    chiplet_compute_power_over_time[chiplet_id][t_idx] += avg_power_in_bin

        # --- Process Communication Power (Activation Communication and Weight Loading) ---
        if dsent_stats:
            sorted_dsent_stats = sorted(dsent_stats, key=lambda x: x['global_time_us'])
            for i, stats_entry in enumerate(sorted_dsent_stats):
                start_time = stats_entry['global_time_us']
                
                # Get the actual communication duration from the stats
                comm_runtime_us = 0.0
                
                # Extract runtime from the communication simulation results
                if 'latency' in stats_entry:
                    # Convert from seconds to microseconds if available
                    if 'total_runtime_s' in stats_entry['latency']:
                        comm_runtime_us = stats_entry['latency']['total_runtime_s'] * 1e6
                    elif 'total_runtime_us' in stats_entry['latency']:
                        comm_runtime_us = stats_entry['latency']['total_runtime_us']
                
                # If still no runtime, fall back to next simulation point (old behavior)
                if comm_runtime_us <= 0:
                    end_time = sorted_dsent_stats[i+1]['global_time_us'] if i + 1 < len(sorted_dsent_stats) else self.global_time_us
                else:
                    # Use actual communication duration
                    end_time = start_time + comm_runtime_us
                    # Ensure we don't exceed simulation time
                    end_time = min(end_time, self.global_time_us)
                
                start_tp_idx = max(0, int(np.floor(start_time / time_step_us)))
                end_tp_idx = min(num_time_points, int(np.ceil(end_time / time_step_us)))

                power_data = stats_entry.get('power', {})
                if not power_data:
                    continue

                # Determine if this is activation communication or weight loading based on phase type
                # For now, we'll distribute power to both activation communication and weight loading
                # In a more sophisticated implementation, we could determine the phase type from the stats
                is_activation_comm = True  # Default assumption
                is_weight_loading = True   # Default assumption
                
                # Router power - distribute to appropriate communication type
                for router_power_info in power_data.get('routers', []):
                    router_id = router_power_info.get('router_id')
                    chiplet_id = router_id + 1
                    
                    total_router_power = 0
                    if 'Total Power' in router_power_info:
                        power_dict = router_power_info.get('Total Power', {})
                        dynamic_power = power_dict.get('Dynamic power (W)', 0.0)
                        leakage_power = power_dict.get('Leakage power (W)', 0.0)
                        total_router_power = dynamic_power + leakage_power

                    if total_router_power > 0:
                        for t_idx in range(start_tp_idx, end_tp_idx):
                            bin_start_time = time_points[t_idx]
                            bin_end_time = bin_start_time + time_step_us
                            overlap_start = max(start_time, bin_start_time)
                            overlap_end = min(end_time, bin_end_time)
                            overlap_duration = max(0, overlap_end - overlap_start)
                            if overlap_duration > 0:
                                avg_power = total_router_power * (overlap_duration / time_step_us)
                                # For now, distribute equally between activation communication and weight loading
                                # In a more sophisticated implementation, we could determine the actual phase type
                                if chiplet_id in chiplet_activation_comm_power_over_time:
                                    chiplet_activation_comm_power_over_time[chiplet_id][t_idx] += avg_power * 0.5
                                if chiplet_id in chiplet_weight_loading_power_over_time:
                                    chiplet_weight_loading_power_over_time[chiplet_id][t_idx] += avg_power * 0.5

                # Link power distribution - distribute to appropriate communication type
                total_link_power = power_data.get('totals', {}).get('link_power_W', 0)
                if total_link_power > 0:
                    chiplet_power_shares = {}
                    main_layer_info = stats_entry.get('main_layer_info')
                    traffic_dict = None
                    if main_layer_info:
                        model_idx, inp_idx, lay_idx = main_layer_info['model_idx'], main_layer_info['input_idx'], main_layer_info['layer_idx']
                        model = self.retired_mapped_models.get(model_idx)
                        if model and hasattr(model, 'layer_communication_sim_traffic'):
                            traffic_dict = model.layer_communication_sim_traffic.get(inp_idx, {}).get(lay_idx)

                    if traffic_dict:
                        chiplet_traffic = {cid: 0 for cid in all_system_chiplets}
                        total_traffic = 0
                        for (src, dst), data in traffic_dict.items():
                            amount = data.get('amount', 0)
                            if src in chiplet_traffic: chiplet_traffic[src] += amount
                            if dst in chiplet_traffic: chiplet_traffic[dst] += amount
                            total_traffic += amount
                        
                        if total_traffic > 0:
                            for cid in all_system_chiplets:
                                share = chiplet_traffic.get(cid, 0) / (2 * total_traffic)
                                power_share = total_link_power * share
                                chiplet_power_shares[cid] = power_share
                        else: # Fallback to even distribution
                            if all_system_chiplets:
                                power_per_chiplet = total_link_power / len(all_system_chiplets)
                                for cid in all_system_chiplets:
                                    chiplet_power_shares[cid] = power_per_chiplet
                    else: # Fallback to even distribution
                        if all_system_chiplets:
                            power_per_chiplet = total_link_power / len(all_system_chiplets)
                            for cid in all_system_chiplets:
                                chiplet_power_shares[cid] = power_per_chiplet
                    
                    # Apply prorated power for each chiplet's share
                    for cid, power_share in chiplet_power_shares.items():
                        if power_share > 0:
                            for t_idx in range(start_tp_idx, end_tp_idx):
                                bin_start_time = time_points[t_idx]
                                bin_end_time = bin_start_time + time_step_us
                                overlap_start = max(start_time, bin_start_time)
                                overlap_end = min(end_time, bin_end_time)
                                overlap_duration = max(0, overlap_end - overlap_start)
                                if overlap_duration > 0:
                                    avg_power = power_share * (overlap_duration / time_step_us)
                                    # For now, distribute equally between activation communication and weight loading
                                    # In a more sophisticated implementation, we could determine the actual phase type
                                    if cid in chiplet_activation_comm_power_over_time:
                                        chiplet_activation_comm_power_over_time[cid][t_idx] += avg_power * 0.5
                                    if cid in chiplet_weight_loading_power_over_time:
                                        chiplet_weight_loading_power_over_time[cid][t_idx] += avg_power * 0.5

        # --- Store results ---
        self.power_time_points = time_points
        self.chiplet_compute_power_over_time = chiplet_compute_power_over_time
        self.chiplet_activation_comm_power_over_time = chiplet_activation_comm_power_over_time
        self.chiplet_weight_loading_power_over_time = chiplet_weight_loading_power_over_time
        
        self.chiplet_total_power_over_time = {
            cid: chiplet_compute_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_activation_comm_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_weight_loading_power_over_time.get(cid, np.zeros(num_time_points))
            for cid in all_system_chiplets
        }
        
        self.system_compute_power_over_time = np.sum(list(chiplet_compute_power_over_time.values()), axis=0) if chiplet_compute_power_over_time else np.zeros(num_time_points)
        self.system_activation_comm_power_over_time = np.sum(list(chiplet_activation_comm_power_over_time.values()), axis=0) if chiplet_activation_comm_power_over_time else np.zeros(num_time_points)
        self.system_weight_loading_power_over_time = np.sum(list(chiplet_weight_loading_power_over_time.values()), axis=0) if chiplet_weight_loading_power_over_time else np.zeros(num_time_points)
        self.system_total_power_over_time = self.system_compute_power_over_time + self.system_activation_comm_power_over_time + self.system_weight_loading_power_over_time
        
        # Create combined communication power attributes for compatibility with plotter
        self.chiplet_communication_power_over_time = {
            cid: chiplet_activation_comm_power_over_time.get(cid, np.zeros(num_time_points)) + 
                 chiplet_weight_loading_power_over_time.get(cid, np.zeros(num_time_points))
            for cid in all_system_chiplets
        }
        self.system_communication_power_over_time = self.system_activation_comm_power_over_time + self.system_weight_loading_power_over_time

    def compute_energy_metrics(self):
        """
        Computes energy consumption metrics based on the computed power profiles.
        This method should be called *after* compute_power_profile.

        It calculates the total energy consumed for compute, communication, and their
        sum, both on a per-chiplet basis and for the entire system.

        The energy is calculated by integrating the power over time, approximated by
        summing the power at each time step multiplied by the time step duration.

        Updates the following instance attributes:
        - self.chiplet_compute_energy_uj
        - self.chiplet_communication_energy_uj
        - self.chiplet_total_energy_uj
        - self.system_compute_energy_uj
        - self.system_communication_energy_uj
        - self.system_total_energy_uj
        """
        # --- Pre-computation Checks ---
        if self.power_time_points is None or self.power_time_step_us is None:
            print("Warning: Power profiles must be computed before energy metrics can be calculated. Call `compute_power_profile` first.")
            return

        if self.power_time_step_us <= 0:
            print(f"Warning: Invalid power time step ({self.power_time_step_us} us). Cannot compute energy metrics.")
            return

        # Energy in microJoules (uJ) is Power (W) * time (us).
        time_step_us = self.power_time_step_us

        # --- Per-Chiplet Energy Calculation ---
        self.chiplet_compute_energy_uj = {}
        if self.chiplet_compute_power_over_time:
            for cid, power_profile in self.chiplet_compute_power_over_time.items():
                self.chiplet_compute_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # Calculate activation communication energy
        self.chiplet_activation_comm_energy_uj = {}
        if self.chiplet_activation_comm_power_over_time:
            for cid, power_profile in self.chiplet_activation_comm_power_over_time.items():
                self.chiplet_activation_comm_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # Calculate weight loading energy
        self.chiplet_weight_loading_energy_uj = {}
        if self.chiplet_weight_loading_power_over_time:
            for cid, power_profile in self.chiplet_weight_loading_power_over_time.items():
                self.chiplet_weight_loading_energy_uj[cid] = np.sum(power_profile) * time_step_us
        
        self.chiplet_total_energy_uj = {}
        if self.chiplet_total_power_over_time:
            for cid, power_profile in self.chiplet_total_power_over_time.items():
                self.chiplet_total_energy_uj[cid] = np.sum(power_profile) * time_step_us

        # --- System-wide Energy Calculation ---
        if self.system_compute_power_over_time is not None:
            self.system_compute_energy_uj = np.sum(self.system_compute_power_over_time) * time_step_us
        else:
            self.system_compute_energy_uj = 0.0

        if self.system_activation_comm_power_over_time is not None:
            self.system_activation_comm_energy_uj = np.sum(self.system_activation_comm_power_over_time) * time_step_us
        else:
            self.system_activation_comm_energy_uj = 0.0

        if self.system_weight_loading_power_over_time is not None:
            self.system_weight_loading_energy_uj = np.sum(self.system_weight_loading_power_over_time) * time_step_us
        else:
            self.system_weight_loading_energy_uj = 0.0

        if self.system_total_power_over_time is not None:
            self.system_total_energy_uj = np.sum(self.system_total_power_over_time) * time_step_us
        else:
            self.system_total_energy_uj = 0.0
            
        # As a check, system total should be sum of compute, activation communication, and weight loading
        # Allow for small floating point discrepancies
        if (self.system_compute_energy_uj is not None and 
            self.system_activation_comm_energy_uj is not None and 
            self.system_weight_loading_energy_uj is not None):
             assert np.isclose(self.system_total_energy_uj, 
                              self.system_compute_energy_uj + 
                              self.system_activation_comm_energy_uj + 
                              self.system_weight_loading_energy_uj), \
                "System total energy does not match the sum of compute, activation communication, and weight loading energy."