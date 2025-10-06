import numpy as np
import random
import math
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx

class ModelMapper:
    """
    Handles mapping models to chiplets.
    
    Responsible for:
    - Generating mappings for models using various mapping algorithms
    - Tracking used crossbars
    - Managing chiplet resources during mapping
    
    Args:
        system (System): The system object containing chiplet information
        mapping_function (str): Name of the mapping function to use, defaults to "nearest_neighbor_v3"
        preference (Dict[str, int]): Preference dictionary for mapping (performance vs energy efficiency)
    """
    
    def __init__(self, 
                 system, 
                 mapping_function="nearest_neighbor_v3", 
                 preference=None):
        """
        Initialize the model mapper.
        
        Args:
            system (System): The system object containing chiplet information
            mapping_function (str): Name of the mapping function to use
            preference (Dict[str, int], optional): Preference dictionary for mapping
        """
        self.system = system
        self.mapping_function = mapping_function
        self.preference = preference or {"performance": 1, "energy_efficiency": 0}
        
        # Get compute chiplets only (exclude I/O chiplets from mapping)
        self.compute_chiplet_indices = self._get_compute_chiplet_indices()
        
        # Precompute all shortest paths between chiplets (this is done once and reused)
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(self.system.chiplet_network))
    
    def generate_mapping(self, model_metrics: Dict[str, Any]) -> Tuple[Optional[List], np.ndarray, Optional[str]]:
        """
        Generate mapping for the model using the selected mapper function.
        Handles mapping atomically: system resources are only updated if the entire model maps successfully.
        
        Args:
            model_metrics (Dict[str, Any]): Model metrics containing layer information
            
        Returns:
            Tuple[Optional[List], np.ndarray, Optional[str]]: 
                - List of layer mappings or None if mapping failed
                - Array of used crossbars per chiplet (only valid if mapping succeeded)
                - Failure reason if mapping failed, None otherwise
        """
        layer_mappings = []
        failure_reason = None
        
        # Get the appropriate mapper function
        mapper_func = self._get_mapper_function()
        
        if not mapper_func:
            print(f"❌ Mapper function {self.mapping_function} not found")
            return None, np.array([]), "INVALID_MAPPER_FUNCTION"
            
        # Get the initial state of the system *before* attempting to map this model
        initial_available_crossbars = self.system.get_available_crossbars_per_chiplet()
        # Create a temporary state to simulate mapping
        temp_available_crossbars = initial_available_crossbars.copy()
        
        # Ensure I/O chiplets have zero available crossbars (they should never be mapped to)
        for idx in range(len(self.system.chiplets)):
            chiplet_id = idx + 1
            if self.system.is_io_chiplet(chiplet_id):
                temp_available_crossbars[idx] = 0
        
        # Process each layer in the model using the temporary state
        for layer_idx, layer_info in enumerate(model_metrics):
            # Map the current layer using the temporary available crossbars
            layer_remaining_crossbars, action, mapping_failed, current_failure_reason = mapper_func(
                layer_info,
                self.system, # Pass system for chiplet info and network topology
                self.preference,
                current_available_crossbars=temp_available_crossbars, # Pass the temporary state
                layer_mappings=layer_mappings,
                shortest_paths=self.shortest_paths  # Pass precomputed shortest paths
            )
            
            if mapping_failed:
                # If any layer fails, the whole mapping fails. Do NOT update the real system state.
                failure_reason = current_failure_reason
                return None, np.array([]), failure_reason
            
            # Update the *temporary* state for the next layer's mapping
            temp_available_crossbars = layer_remaining_crossbars
            
            # Store the mapping result (action percentages)
            layer_mappings.append((layer_idx, [(i+1, pct) for i, pct in enumerate(action) if pct > 0]))
        
        # If the loop completes, the entire model was successfully mapped (in the temporary state)
        
        # Calculate the total crossbars used based on the initial and final *temporary* state
        used_crossbars = initial_available_crossbars - temp_available_crossbars
        
        # Now, commit the changes to the *real* system state
        update_success = self.system.update_crossbar_availability(temp_available_crossbars)
        
        if not update_success:
            # This indicates an internal error, as the mapping seemed successful
            failure_reason = "SYSTEM_UPDATE_FAILURE"
            print(f"❌ CRITICAL: Failed to update system crossbar availability after successful temporary mapping.")
            # Return failure, although the real system state might be inconsistent now
            return None, np.array([]), failure_reason
            
        # Return the successful mapping and the calculated used crossbars
        return layer_mappings, used_crossbars, None
    
    def _get_mapper_function(self):
        """
        Get the mapper function based on the mapping_function attribute.
        
        Returns:
            function: Mapper function to use for mapping
        """
        if self.mapping_function == "nearest_neighbor_v3":
            return self._nearest_neighbor_mapper_v3
        elif self.mapping_function == "random_mapper":
            return self._random_mapper
        else:
            try:
                # Try to dynamically get the mapper function if it's a custom method
                mapper_func = getattr(self, self.mapping_function)
                return mapper_func
            except AttributeError:
                print(f"❌ Error: Mapper function '{self.mapping_function}' not found as a method in ModelMapper.")
                return None
    
    def _get_compute_chiplet_indices(self):
        """
        Get indices of compute chiplets only (excluding I/O chiplets).
        
        Returns:
            List[int]: List of compute chiplet indices (0-based)
        """
        compute_indices = []
        for idx, chiplet in enumerate(self.system.chiplets):
            # Check if this is a compute chiplet (not an I/O chiplet)
            chiplet_id = idx + 1  # Convert to 1-based chiplet ID
            if not self.system.is_io_chiplet(chiplet_id):
                compute_indices.append(idx)
        

        if len(compute_indices) < len(self.system.chiplets):
            io_count = len(self.system.chiplets) - len(compute_indices)
            print(f"✓ Excluding {io_count} I/O chiplet(s) from mapping consideration")
        
        return compute_indices

    def _calculate_layer_requirements(self, layer_info, chiplet):
        """
        Calculate chiplet-specific requirements for the layer.
        Args:
            layer_info (list): Information about the layer. [name, layer_type, input_activation, total_macs, num_weights, filter_size, out_channels]
            or [name, layer_type, input_activation, total_macs, num_weights, in_features, out_features]
            chiplet (Chiplet): Chiplet object with specifications.
        Returns:
            total_crosssbars_layer (int): Total number of crossbars required for the layer.
        """
        # Extract chiplet-specific parameters using dot notation
        CROSSBAR_ROWS = chiplet.crossbar_rows
        CROSSBAR_COLUMNS = chiplet.crossbar_columns
        BITS_PER_CELL = chiplet.bits_per_cell
        BITS_PER_WEIGHT = chiplet.bits_per_weight # 8 all the time

        if layer_info['layer_type'] == 'conv':
            filter_size = layer_info['filter_size'] # filter_size = in_channels * (kernel_size ** 2)
            total_filters = layer_info['out_channels']

            col_required = total_filters * BITS_PER_WEIGHT / BITS_PER_CELL
            row_required = filter_size

            col_crossbars = math.ceil(col_required/CROSSBAR_COLUMNS)
            row_crossbars = math.ceil(row_required/CROSSBAR_ROWS)

            total_crossbars_layer = col_crossbars * row_crossbars

        elif layer_info['layer_type'] == 'fc':
            col_required = layer_info['out_features'] * BITS_PER_WEIGHT/BITS_PER_CELL
            row_required = layer_info['in_features']

            col_crossbars = math.ceil(col_required/CROSSBAR_COLUMNS)
            row_crossbars = math.ceil(row_required/CROSSBAR_ROWS)

            total_crossbars_layer = col_crossbars * row_crossbars

        elif layer_info['layer_type'] == 'self_attention':
            # For self-attention layers, treat as matrix multiplication similar to FC
            # Main operations: Q, K, V projections and output projection
            # Each projection: [seq_length, embed_dim] × [embed_dim, embed_dim]
            embed_dim = layer_info['out_channels']  # Total embedding dimension
            
            col_required = embed_dim * BITS_PER_WEIGHT / BITS_PER_CELL
            row_required = embed_dim  # Input embedding dimension

            col_crossbars = math.ceil(col_required / CROSSBAR_COLUMNS)
            row_crossbars = math.ceil(row_required / CROSSBAR_ROWS)

            total_crossbars_layer = col_crossbars * row_crossbars
        
        else:
            print(f"Error: Unknown layer type {layer_info['layer_type']}")
            total_crossbars_layer = 0

        return total_crossbars_layer

    def _random_mapper(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
        """
        Randomly maps model layers to available crossbars across chiplets.
        Note: This is a wrapper to maintain a consistent signature with other mappers.
        """
        mapping_failed = False
        failure_reason = None
        
        chiplets = system.chiplets
        num_chiplets = len(chiplets)
        
        remaining_crossbars = current_available_crossbars.copy()

        layer_name = model_layer_info['name']
        crossbars_required = model_layer_info['crossbars_required']
        action = [0] * num_chiplets
        resources_remaining = crossbars_required
        percentage_remaining = 100
        
        # Only consider compute chiplets (exclude I/O chiplets)
        chiplet_indices = [idx for idx in range(num_chiplets) 
                          if not self.system.is_io_chiplet(idx + 1)]
        random.shuffle(chiplet_indices)

        for chiplet_idx in chiplet_indices:
            if resources_remaining <= 0:
                break
            
            available_cb = remaining_crossbars[chiplet_idx]
            if available_cb <= 0:
                continue
            
            crossbars_needed = self._calculate_layer_requirements(model_layer_info, chiplets[chiplet_idx])
            
            if crossbars_required > 0:
                crossbars_needed_for_resources = math.ceil(resources_remaining * crossbars_needed / crossbars_required)
            else:
                crossbars_needed_for_resources = 0

            if available_cb >= crossbars_needed_for_resources:
                remaining_crossbars[chiplet_idx] -= crossbars_needed_for_resources
                alloc_percentage = percentage_remaining
                percentage_remaining = 0
                resources_remaining = 0
            else:
                if crossbars_needed > 0:
                    alloc_percentage = available_cb * percentage_remaining / crossbars_needed
                else:
                    alloc_percentage = 0
                remaining_crossbars[chiplet_idx] = 0
                percentage_remaining -= alloc_percentage
                resources_remaining -= math.ceil(alloc_percentage * crossbars_required / 100)
            
            action[chiplet_idx] = alloc_percentage

        if (resources_remaining - 10) > 0:
            print(f"⚠️ Layer '{layer_name}' allocation postponed: {resources_remaining} resources could not be allocated.")
            mapping_failed = True
            failure_reason = "INSUFFICIENT_MEMORY"
            return current_available_crossbars, None, mapping_failed, failure_reason

        sum_action = sum(action)
        if sum_action > 0 and not math.isclose(sum_action, 100, rel_tol=1):
             print(f"Warning: Layer '{layer_name}' action sum is {sum_action}, not 100. Action: {action}")

        return remaining_crossbars, action, mapping_failed, failure_reason

    def _nearest_neighbor_mapper_v3(self, model_layer_info, system, preference, current_available_crossbars, layer_mappings=None, shortest_paths=None):
        """
        Maps layers to chiplets based on nearest neighbor mapping.
        """
        chiplet_network = system.chiplet_network
        chiplets = system.chiplets
        
        mapping_failed = False
        failure_reason = None
        num_chiplets = len(chiplets)
        remaining_crossbars = current_available_crossbars.copy()

        if shortest_paths is None:
            shortest_paths = dict(nx.all_pairs_shortest_path_length(chiplet_network))
        
        active_metric = next((metric for metric, active in preference.items() if active), "performance")
            
        layer_name = model_layer_info['name']
        crossbars_required = model_layer_info['crossbars_required']
        action = [0] * num_chiplets
        resources_remaining = crossbars_required
        percentage_remaining = 100

        occupancy_percentages = []
        avail_memory_list = []
        
        for idx, chiplet in enumerate(chiplets):
            # Skip I/O chiplets
            if self.system.is_io_chiplet(idx + 1):
                avail_memory_list.append(0)
                occupancy_percentages.append(100)
            else:
                avail_memory = remaining_crossbars[idx] * chiplet.memory_per_crossbar
                total_memory = chiplet.get_total_memory()
                occupancy = (total_memory - avail_memory) / total_memory * 100 if total_memory > 0 else 100
                avail_memory_list.append(avail_memory)
                occupancy_percentages.append(occupancy)
            
        if not layer_mappings:
            available_chiplet_ids = [idx + 1 for idx, cb in enumerate(remaining_crossbars) 
                                    if cb > 0 and not self.system.is_io_chiplet(idx + 1)]
            if not available_chiplet_ids:
                return current_available_crossbars, None, True, "NO_AVAILABLE_CHIPLETS"

            chiplet_info = []
            for chiplet_id in available_chiplet_ids:
                idx = chiplet_id - 1
                metric_value = chiplet_network.nodes[chiplet_id].get(active_metric, 0)
                chiplet_info.append((chiplet_id, avail_memory_list[idx], metric_value))

            chiplet_info.sort(key=lambda x: (-x[1], -x[2]))
            
            sorted_chiplets_ids = [chiplet_id for chiplet_id, _, _ in chiplet_info]
            starting_chiplet = sorted_chiplets_ids[0]
        else:
            last_layer_mapping = layer_mappings[-1][1]
            last_chiplet = max(last_layer_mapping, key=lambda x: x[1])[0]
            starting_chiplet = last_chiplet
            
        distances = shortest_paths.get(starting_chiplet, {})
        if not distances and num_chiplets > 1:
            return current_available_crossbars, None, True, "NO_CHIPLET_CONNECTIONS"

        available_chiplet_ids = [idx + 1 for idx, cb in enumerate(remaining_crossbars) 
                                if cb > 0 and not self.system.is_io_chiplet(idx + 1)]
        
        chiplet_info = []
        for chiplet_id in available_chiplet_ids:
            idx = chiplet_id - 1
            distance = distances.get(chiplet_id, math.inf)
            metric_value = chiplet_network.nodes[chiplet_id].get(active_metric, 0)
            chiplet_info.append((chiplet_id, distance, avail_memory_list[idx], metric_value))

        chiplet_info.sort(key=lambda x: (x[1], -x[2], -x[3]))
        
        sorted_chiplets_ids = [chiplet_id for chiplet_id, _, _, _ in chiplet_info]
        
        for chiplet_id in sorted_chiplets_ids:
            if resources_remaining <= 0:
                break
            chiplet_idx = chiplet_id - 1
            available_cb = remaining_crossbars[chiplet_idx]
            if available_cb <= 0:
                continue

            crossbars_needed = self._calculate_layer_requirements(model_layer_info, chiplets[chiplet_idx])
            if crossbars_needed <= 0: continue

            if crossbars_required > 0:
                crossbars_needed_for_resources = math.ceil(resources_remaining * crossbars_needed / crossbars_required)
            else:
                crossbars_needed_for_resources = 0
            
            if available_cb >= crossbars_needed_for_resources:
                remaining_crossbars[chiplet_idx] -= crossbars_needed_for_resources
                alloc_percentage = percentage_remaining
                percentage_remaining = 0
                resources_remaining = 0
            else:
                alloc_percentage = available_cb * 100 / crossbars_needed
                remaining_crossbars[chiplet_idx] = 0
                percentage_remaining -= alloc_percentage
                resources_remaining -= math.ceil(alloc_percentage * crossbars_required / 100)
            
            action[chiplet_idx] += alloc_percentage

        if (resources_remaining - 10) > 0:
            return current_available_crossbars, None, True, "INSUFFICIENT_MEMORY"

        sum_action = sum(action)
        if sum_action > 0 and not math.isclose(sum_action, 100, rel_tol=1):
             print(f"Warning: Layer '{layer_name}' action sum is {sum_action}, not 100. Action: {action}")
        
        return remaining_crossbars, action, False, None
    
    def set_preference(self, performance: float = 1.0, energy_efficiency: float = 0.0):
        """
        Set the preference for mapping between performance and energy efficiency.
        
        Args:
            performance (float): Weight for performance preference (0.0 to 1.0)
            energy_efficiency (float): Weight for energy efficiency preference (0.0 to 1.0)
        """
        # Normalize the weights to ensure they sum to 1
        total = performance + energy_efficiency
        if total > 0:
            self.preference = {
                "performance": performance / total,
                "energy_efficiency": energy_efficiency / total
            }
        else:
            # Default to performance if both are 0
            self.preference = {"performance": 1, "energy_efficiency": 0}
            
