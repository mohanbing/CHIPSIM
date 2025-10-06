import yaml
import numpy as np
import networkx as nx
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .chiplet import Chiplet
from assets.chiplet_specs.chiplet_params import CHIPLET_TYPES

# load material properties from yaml file
def load_dict_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        dict = yaml.safe_load(f)
    return dict

class System:
    """
    Models and manages a cluster of chiplets, tracking crossbar availability.
    """

    def __init__(
        self,
        chiplet_mapping_file,
        adj_matrix_file,
    ):
        """
        Initializes the SystemState with chiplet configurations and simulation parameters.

        Parameters:
            chiplet_mapping_file (str): YAML file containing chiplet mapping.
            adj_matrix_file (str): File containing the adjacency matrix for chiplet connections.
        """
        
        # Load chiplet mapping from YAML
        self.chiplet_mapping = load_dict_yaml(chiplet_mapping_file)
        self.num_chiplets = len(self.chiplet_mapping)
        self.chiplets = [
            Chiplet(chiplet_id=i + 1, chiplet_type=self.chiplet_mapping[i + 1])
            for i in range(self.num_chiplets)
        ]
        
        # Initialize the total memory in the system
        self.system_memory = 0
        
        # Get the total amount of memory in the system, in bits
        for chiplet in self.chiplets:
            self.system_memory += chiplet.get_total_memory()

        self.get_memory_per_crossbar()
        
        # Load adjacency matrix
        self.adj_matrix = np.loadtxt(adj_matrix_file)
        
        # Build the chiplet network from the adjacency matrix
        self.chiplet_network = self.build_chiplet_network()
        
        # Identify I/O and compute chiplets
        self.io_chiplet_ids = []
        self.compute_chiplet_ids = []
        self._identify_chiplet_types()
        
    def build_chiplet_network(self):
        """
        Build the chiplet network topology from the adjacency matrix file and chiplet mapping,
        including energy efficiency and performance metrics for each chiplet.

        Args:
            adj_matrix_file (str): Filename of the adjacency matrix.
            chiplet_mapping (dict): Mapping from chiplet numbers to chiplet types.

        Returns:
            networkx.Graph: Graph representing the chiplet network with additional metrics.
        """
        # Convert CHIPLET_TYPES list to a dictionary for efficient lookup
        CHIPLET_TYPES_DICT = {chiplet['name']: chiplet for chiplet in CHIPLET_TYPES}

        # Create a graph
        G = nx.Graph()

        # Add nodes with chiplet types and additional metrics
        for chiplet_num in range(self.num_chiplets):
            chiplet_id = chiplet_num + 1  # Chiplet numbers start from 1
            chiplet_type = self.chiplet_mapping.get(chiplet_id, 'Unknown')

            # Retrieve metrics from CHIPLET_TYPES_DICT
            chiplet_info = CHIPLET_TYPES_DICT.get(chiplet_type, {})
            performance = chiplet_info.get('tops', 0)  # Default to 0 if not found
            energy_efficiency = chiplet_info.get('energy_per_mac', 0)  # Default to 0 if not found

            # Add node with all attributes
            G.add_node(
                chiplet_id,
                chiplet_type=chiplet_type,
                performance=performance,
                energy_efficiency=energy_efficiency,
            )
        
        total_nodes = self.num_chiplets

        # Add edges based on adjacency matrix
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                if self.adj_matrix[i, j] == 1:
                    G.add_edge(i + 1, j + 1)  # Adjust index to start from 1

        return G
    
    def _identify_chiplet_types(self):
        """Identify and categorize I/O vs compute chiplets based on their type"""
        for chiplet in self.chiplets:
            if chiplet.type == "IO":
                self.io_chiplet_ids.append(chiplet.id)
            else:
                self.compute_chiplet_ids.append(chiplet.id)
    
    def is_io_chiplet(self, chiplet_id: int) -> bool:
        """Check if a chiplet is an I/O chiplet"""
        return chiplet_id in self.io_chiplet_ids
    
    def is_compute_chiplet(self, chiplet_id: int) -> bool:
        """Check if a chiplet is a compute chiplet"""
        return chiplet_id in self.compute_chiplet_ids
    
    def get_edge_chiplets(self) -> list:
        """Get compute chiplets at the edge of the mesh (connected to I/O)"""
        edge_chiplets = []
        for io_id in self.io_chiplet_ids:
            # Find all compute chiplets connected to this I/O chiplet
            if io_id in self.chiplet_network:
                neighbors = list(self.chiplet_network.neighbors(io_id))
                edge_chiplets.extend([n for n in neighbors if self.is_compute_chiplet(n)])
        return list(set(edge_chiplets))
    
    def get_io_chiplet_for_compute(self, compute_chiplet_id: int) -> int:
        """
        Determine which I/O chiplet should serve a compute chiplet
        based on network distance (shortest path).
        
        Args:
            compute_chiplet_id: ID of the compute chiplet
            
        Returns:
            ID of the closest I/O chiplet, or None if no I/O chiplets exist
        """
        if not self.io_chiplet_ids:
            return None
            
        min_distance = float('inf')
        closest_io = None
        
        for io_id in self.io_chiplet_ids:
            distance = self.get_path_length_between_chiplets(io_id, compute_chiplet_id)
            if distance != -1 and distance < min_distance:
                min_distance = distance
                closest_io = io_id
                
        return closest_io
    
    def get_io_chiplet_connections(self) -> dict:
        """
        Get mapping of I/O chiplets to their directly connected compute chiplets.
        
        Returns:
            Dictionary mapping I/O chiplet IDs to lists of connected compute chiplet IDs
        """
        connections = {}
        for io_id in self.io_chiplet_ids:
            if io_id in self.chiplet_network:
                connections[io_id] = [
                    n for n in self.chiplet_network.neighbors(io_id) 
                    if self.is_compute_chiplet(n)
                ]
            else:
                connections[io_id] = []
        return connections

    def get_available_memory_in_system(self):
        """
        Retrieves the current number of available memory in int8 units in each chiplet.

        Returns:
            int: Total available memory in the system.
        """
        available_memory = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            available_memory[chiplet_id - 1] = chiplet.get_available_memory()
        
        return available_memory

    def get_available_crossbars_per_chiplet(self):
        """
        Retrieves the current number of available crossbars in each chiplet.

        Returns:
            numpy.ndarray: Array where each element corresponds to the number of available crossbars in a chiplet.
        """
        crossbar_availability = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            available_crossbars = chiplet.get_available_crossbars()
            crossbar_availability[chiplet_id - 1] = available_crossbars

        return crossbar_availability
    
    def get_memory_per_crossbar(self):
        """
        Retrieves the memory per crossbar in the system.

        Returns:
            numpy.ndarray: Array where each element corresponds to the memory per crossbar in a chiplet.
        """
        self.memory_per_crossbar = np.zeros(self.num_chiplets, dtype=int)
        for chiplet in self.chiplets:
            chiplet_id = chiplet.id
            self.memory_per_crossbar[chiplet_id - 1] = chiplet.memory_per_crossbar        
    
    def get_total_system_memory(self):
        """
        Get the total memory capacity of the entire system (all chiplets).
        
        Returns:
            float: Total system memory in weight units
        """
        return self.system_memory
    
    def get_model_memory_requirement(self, model_metrics):
        """
        Calculate the total memory requirement for a model.
        
        Args:
            model_metrics (list): A list of dictionaries, where each dict describes a layer.
            
        Returns:
            float: Total memory requirement in weight units
        """
        # Use crossbars_required for memory requirement calculation
        return sum(layer['crossbars_required'] for layer in model_metrics)
    
    def can_model_ever_fit(self, model_metrics):
        """
        Check if a model can ever fit in the system by comparing against total system memory.
        This is a definitive check - if this returns False, the model will never fit.
        
        Args:
            model_metrics (list): A list of dictionaries, where each dict describes a layer.
            
        Returns:
            tuple: (bool, str or None). True if model can potentially fit,
                   False with detailed error message if it can never fit.
        """
        total_required_weights = self.get_model_memory_requirement(model_metrics)
        total_system_memory = self.get_total_system_memory()
        
        if total_required_weights > total_system_memory:
            return False, f"MODEL_TOO_LARGE_FOR_SYSTEM"
        
        return True, None

    def can_map_model(self, model_metrics):
        """
        Performs a pre-check to see if there is enough total memory
        in the system for an entire model. This does not account for
        fragmentation, so a successful check does not guarantee a successful mapping.

        Args:
            model_metrics (list): A list of dictionaries, where each dict describes a layer.

        Returns:
            tuple: (bool, str or None). True if it can technically be mapped,
                   False with a reason string if not.
        """
        # Use crossbars_required for memory requirement
        total_required_weights = sum(layer['crossbars_required'] for layer in model_metrics)

        available_crossbars = self.get_available_crossbars_per_chiplet()
        total_available_memory = (self.memory_per_crossbar * available_crossbars).sum()

        if total_required_weights > total_available_memory:
            return False, "INSUFFICIENT_MEMORY"

        return True, None

    def update_crossbar_availability(self, remaining_crossbars):
        """
        Updates the available crossbars in each chiplet based on mapping results.
        
        Args:
            remaining_crossbars (np.array): Array indicating remaining available crossbars per chiplet.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        if len(remaining_crossbars) != self.num_chiplets:
            print(f"System Class Error: Expected {self.num_chiplets} values in remaining_crossbars array, got {len(remaining_crossbars)}")
            exit(1)
        
        # Update the available crossbars for each chiplet
        for i, chiplet in enumerate(self.chiplets):
            # Ensure that the new available crossbars are not greater than the total crossbars
            if remaining_crossbars[i] > chiplet.total_crossbars:
                raise ValueError(f"System Class Error: Chiplet {chiplet.id} would have more available crossbars than it has total crossbars")
                
            # Set the new availability directly
            chiplet.set_available_crossbars(remaining_crossbars[i])
        
        return True
        
    def get_path_length_between_chiplets(self, chiplet_id1, chiplet_id2):
        """
        Calculate the number of links that need to be traversed between two chiplets
        in the chiplet network (shortest path).
        
        Args:
            chiplet_id1 (int): ID of the first chiplet
            chiplet_id2 (int): ID of the second chiplet
            
        Returns:
            int: Number of links between the chiplets, or -1 if no path exists
        """
        # Check if chiplet IDs are valid
        if chiplet_id1 < 1 or chiplet_id1 > self.num_chiplets or chiplet_id2 < 1 or chiplet_id2 > self.num_chiplets:
            print(f"Error: Invalid chiplet ID. Valid IDs are 1-{self.num_chiplets}")
            return -1
            
        # If the chiplets are the same, no links need to be traversed
        if chiplet_id1 == chiplet_id2:
            return 0
            
        try:
            # Use NetworkX to calculate the shortest path length
            path_length = nx.shortest_path_length(self.chiplet_network, source=chiplet_id1, target=chiplet_id2)
            return path_length
        except nx.NetworkXNoPath:
            # No path exists between the chiplets
            print(f"No path exists between chiplet {chiplet_id1} and chiplet {chiplet_id2}")
            return -1
        except Exception as e:
            print(f"Error calculating path length: {str(e)}")
            return -1