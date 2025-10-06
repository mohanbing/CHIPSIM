import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from assets.chiplet_specs.chiplet_params import CHIPLET_TYPES

class Chiplet:
    """
    Represents a single chiplet with specific hardware attributes.
    """

    def __init__(self, 
                 chiplet_id, 
                 chiplet_type):
        """
        Initializes a Chiplet with given hardware attributes.

        Parameters:
            chiplet_id (int): Unique identifier for the chiplet.
            chiplet_type (str): Name/type of the chiplet.
        """
        self.id = chiplet_id
        self.type = chiplet_type
        chiplet_spec = next((c for c in CHIPLET_TYPES if c['name'] == chiplet_type), None)

        if chiplet_spec is None:
            error_msg = f"Error: Chiplet type '{chiplet_type}' not found in CHIPLET_TYPES."
            print(error_msg)
            raise ValueError(error_msg)

        # Initialize chiplet specifications
        for key, value in chiplet_spec.items():
            setattr(self, key, value)

        # Calculate the memory per crossbar (handle I/O chiplets with no crossbars)
        if self.bits_per_weight > 0 and self.crossbar_rows > 0 and self.crossbar_columns > 0:
            self.memory_per_crossbar = self.crossbar_rows * self.crossbar_columns * self.bits_per_cell / self.bits_per_weight
        else:
            self.memory_per_crossbar = 0  # I/O chiplets have no memory

        # Calculate total crossbars
        self.total_crossbars = self.tiles_per_chiplet * self.crossbars_per_tile

        # Initialize crossbar availability as a simple counter instead of a 2D array
        self.crossbars_available = self.total_crossbars  # All crossbars are initially available

    def get_available_crossbars(self):
        """
        Returns the number of available crossbars in the chiplet.

        Returns:
            int: Number of available crossbars.
        """
        return self.crossbars_available

    def get_available_memory(self):
        """
        Returns the number of available memory units in the chiplet.

        Returns:
            int: Number of available memory units.
        """
        return self.get_available_crossbars() * self.memory_per_crossbar
    
    def get_total_memory(self):
        """
        Returns the total amount of weights that the chiplet can hold.

        Returns:
            float: maximum number of weights that can be stored.
        """
        return self.crossbars_per_tile * self.tiles_per_chiplet * self.memory_per_crossbar

    def set_available_crossbars(self, available_crossbars):
        """
        Sets the number of available crossbars in this chiplet.
        
        Args:
            available_crossbars (int): Number of available crossbars to set.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if available_crossbars < 0:
            error_msg = f"Error: Cannot set negative number of available crossbars ({available_crossbars})"
            print(error_msg)
            return False
        
        if available_crossbars > self.total_crossbars:
            error_msg = f"Error: Cannot set available crossbars ({available_crossbars}) greater than total crossbars ({self.total_crossbars})"
            print(error_msg)
            return False
        
        # Simply set the available crossbars to the specified value
        self.crossbars_available = available_crossbars
        return True 