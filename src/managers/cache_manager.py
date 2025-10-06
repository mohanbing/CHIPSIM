import os
import json
import numpy as np
import hashlib

class CacheManager:
    """
    Handles caching of simulation results to avoid redundant computations.
    """
    def __init__(self, cache_file, clear_cache=False):
        """
        Initialize the cache manager.
        
        Args:
            cache_file (str): Path to the cache file
            clear_cache (bool): Whether to clear the existing cache
        """
        self.cache_file = cache_file
        self.clear_cache = clear_cache
        self.cache = self._load_cache()
        
    def _load_cache(self):
        """
        Load the cache from file or create an empty cache.
        
        Returns:
            dict: The cache data
        """
        try:
            # If clear_cache is True, delete existing cache file and return empty cache
            if self.clear_cache:
                if os.path.exists(self.cache_file):
                    print(f"🗑️ Clearing cache at {self.cache_file}")
                    os.remove(self.cache_file)
                return {}
                
            # Otherwise load the cache if it exists
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    loaded_cache = json.load(f)
                    # Convert string keys back to tuples where appropriate
                    return self._deserialize_cache(loaded_cache)
            else:
                # Create the directory if it doesn't exist
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                # Return an empty cache
                return {}
        except Exception as e:
            print(f"⚠️ Warning: Failed to load cache from {self.cache_file}: {e}")
            return {}
    
    def save_cache(self):
        """
        Save the cache to file.
        """
        try:
            # Convert cache to JSON-serializable format
            serializable_cache = self._make_serializable(self.cache)
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save cache to {self.cache_file}: {e}")
    
    def _make_serializable(self, obj):
        """
        Convert an object to a JSON-serializable format.
        Handles tuples as keys by converting them to strings.
        """
        if isinstance(obj, dict):
            serializable = {}
            for key, value in obj.items():
                # Convert tuple keys to strings
                if isinstance(key, tuple):
                    new_key = str(key)
                else:
                    new_key = key
                serializable[new_key] = self._make_serializable(value)
            return serializable
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _deserialize_cache(self, obj):
        """
        Deserialize cache data, converting string representations of tuples back to tuples.
        This is a simple implementation that doesn't try to parse all strings.
        """
        # For now, just return the object as-is since the cache typically uses string keys
        # If needed, implement tuple string parsing here
        return obj

    def compute_cache_key(self, **params):
        """
        Compute a unique hash key for the provided parameters.
        
        Args:
            **params: Variable keyword arguments to include in the key
            
        Returns:
            str: Hash key for the parameters
        """
        # Create a copy of params that we can modify
        serializable_params = {}
        
        # Convert numpy arrays to lists for serialization
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            else:
                serializable_params[key] = value
        
        # Create a string with all parameters
        param_str = json.dumps(serializable_params, sort_keys=True)
        
        # Compute the SHA-256 hash
        hash_obj = hashlib.sha256(param_str.encode())
        return hash_obj.hexdigest()
    
    def get_result(self, key):
        """
        Get a result from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            object: Cached result or None if not found
        """
        return self.cache.get(key)
    
    def has_result(self, key):
        """
        Check if a result exists in the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if the key exists in the cache
        """
        return key in self.cache
    
    def store_result(self, key, result):
        """
        Store a result in the cache.
        
        Args:
            key (str): Cache key
            result (object): Result to store
        """
        self.cache[key] = result
        self.save_cache() 