"""
Stats Collector Module

Generic utility for collecting and persisting simulation statistics to prevent memory overflow.
Supports batched writes to JSONL format for efficient streaming analysis.
"""

import os
import json
from typing import List, Dict, Any


class StatsCollector:
    """
    Generic collector for simulation statistics with batched file persistence.
    
    This class manages a buffer of statistics entries and periodically dumps them
    to a JSONL file to prevent memory overflow during long simulations.
    
    Args:
        stats_file_path (str): Path to the output JSONL file
        dump_threshold (int): Number of entries to buffer before auto-dumping
        stats_type (str): Type of statistics being collected (e.g., "dsent", "timing")
        auto_initialize_file (bool): Whether to clear/initialize the file on creation
    """
    
    def __init__(self, 
                 stats_file_path: str, 
                 dump_threshold: int = 1,
                 stats_type: str = "generic",
                 auto_initialize_file: bool = True):
        self.stats_file_path = stats_file_path
        self.dump_threshold = dump_threshold
        self.stats_type = stats_type
        self.stats_buffer: List[Dict[str, Any]] = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
        
        # Initialize/clear file if requested
        if auto_initialize_file:
            self._initialize_file()
    
    def _initialize_file(self):
        """Clear/initialize the stats file at the start"""
        try:
            with open(self.stats_file_path, 'w') as f:
                pass  # Just clear the file
        except Exception as e:
            print(f"⚠️ WARNING: Could not initialize stats file {self.stats_file_path}: {e}")
    
    def add_stats(self, stats_entry: Dict[str, Any]):
        """
        Add a statistics entry to the buffer.
        Automatically dumps to file if threshold is reached.
        
        Args:
            stats_entry (dict): Statistics entry to add
        """
        self.stats_buffer.append(stats_entry)
        
        # Auto-dump if threshold reached
        if len(self.stats_buffer) >= self.dump_threshold:
            self.dump_stats()
    
    def dump_stats(self):
        """
        Dump the collected stats to file and clear the buffer.
        Appends to the JSONL file (one JSON object per line).
        """
        if not self.stats_buffer:
            return
        
        try:
            with open(self.stats_file_path, 'a') as f:
                for entry in self.stats_buffer:
                    f.write(json.dumps(entry) + '\n')
            
            if len(self.stats_buffer) >= 10:  # Only print for larger dumps
                print(f"INFO: Dumped {len(self.stats_buffer)} {self.stats_type} stats to {self.stats_file_path}")
            
            # Clear the buffer after successful dump
            self.stats_buffer.clear()
            
        except Exception as e:
            print(f"❌ ERROR: Could not dump {self.stats_type} stats to file: {e}")
    
    def finalize(self):
        """
        Dump any remaining stats in the buffer.
        Should be called at the end of simulation.
        """
        self.dump_stats()
    
    def get_buffer_size(self) -> int:
        """
        Get the current number of entries in the buffer.
        
        Returns:
            int: Number of buffered entries
        """
        return len(self.stats_buffer)
    
    def clear_buffer(self):
        """Clear the buffer without dumping (use with caution)"""
        self.stats_buffer.clear()

