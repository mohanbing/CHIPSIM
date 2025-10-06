"""
generate_network_definition.py

Generates complete network definitions (structure + histograms) for PyTorch models.
Saves outputs in /assets/DNN_models/{network_name} with each layer in its own YAML file.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gc
import os
import time
import argparse
from collections import defaultdict, OrderedDict
import sys

# RuntimeTracker class for performance monitoring
class RuntimeTracker:
    def __init__(self):
        self.timers = {}
        self.active_timers = {}
        self.parent_stack = []
    
    def start(self, name):
        """Start a timer with the given name"""
        parent = self.parent_stack[-1] if self.parent_stack else None
        timer_key = (parent, name) if parent else name
        self.active_timers[timer_key] = time.time()
        self.parent_stack.append(timer_key)
        return timer_key
    
    def stop(self, key=None):
        """Stop a timer and record its duration"""
        if not key:
            key = self.parent_stack.pop() if self.parent_stack else None
            if not key:
                return 0
        
        if key in self.active_timers:
            elapsed = time.time() - self.active_timers[key]
            if key not in self.timers:
                self.timers[key] = []
            self.timers[key].append(elapsed)
            del self.active_timers[key]
            return elapsed
        return 0
    
    def get_total(self, key):
        """Get total time for a specific timer"""
        return sum(self.timers.get(key, [0]))
    
    def summary(self):
        """Print summary of all timers"""
        print("\n--- Runtime Summary ---")
        # Group timers by parent
        by_parent = defaultdict(list)
        for key in self.timers:
            if isinstance(key, tuple):
                parent, name = key
                by_parent[parent].append((name, sum(self.timers[key])))
            else:
                by_parent[None].append((key, sum(self.timers[key])))
        
        # Print top-level timers
        for name, total_time in sorted(by_parent.get(None, []), key=lambda x: x[1], reverse=True):
            print(f"{name}: {total_time:.2f}s")
            # Print children
            for child_name, child_time in sorted(by_parent.get(name, []), key=lambda x: x[1], reverse=True):
                print(f"  └─ {child_name}: {child_time:.2f}s ({child_time/total_time*100:.1f}%)")
        
        return self.timers

# Initialize global runtime tracker
runtime_tracker = RuntimeTracker()

# Default values from the problem_base template
DEFAULTS = {
    'C': 1, 'M': 1, 'P': 1, 'Q': 1,
    'R': 1, 'S': 1,
    'Hstride': 1, 'Wstride': 1,
    'Hdilation': 1, 'Wdilation': 1,
    'G': 1,
    'N': 'BATCH_SIZE',
}

# Try to import tqdm for progress bar, use a simple fallback if not available
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False

# Helper for Conv2d structure extraction
def conv2d_instance(layer, input_shape):
    C, H, W = input_shape
    M = layer.out_channels
    R, S = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    Hstride, Wstride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
    Hdilation, Wdilation = layer.dilation if isinstance(layer.dilation, tuple) else (layer.dilation, layer.dilation)
    G = layer.groups

    P = (H + 2*layer.padding[0] - Hdilation*(R - 1) - 1) // Hstride + 1
    Q = (W + 2*layer.padding[1] - Wdilation*(S - 1) - 1) // Wstride + 1

    instance = {
        'C': C, 'M': M, 'P': P, 'Q': Q,
        'R': R, 'S': S,
        'Hstride': Hstride, 'Wstride': Wstride,
        'Hdilation': Hdilation, 'Wdilation': Wdilation,
        'G': G,
    }

    return instance, (M, P, Q)  # return updated shape

# Helper for Linear structure extraction
def linear_instance(layer, input_shape):
    if len(input_shape) > 1:
        # Flatten if we have spatial dimensions
        C = input_shape[0] * input_shape[1] * input_shape[2] if len(input_shape) == 3 else input_shape[0]
    else:
        C = input_shape[0]
    
    M = layer.out_features

    instance = {
        'C': C,
        'M': M
    }

    return instance, (M,)  # return new shape (no spatial dims)

def is_norm_layer(layer):
    return isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))

# Helper for pool layer structure extraction
def pool_instance(layer, input_shape):
    C, H, W = input_shape
    
    # Get kernel size
    kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
    R, S = kernel_size
    
    # Get stride
    stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
    Hstride, Wstride = stride
    
    # Get padding
    padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
    
    # Calculate output dimensions
    P = (H + 2*padding[0] - R) // Hstride + 1
    Q = (W + 2*padding[1] - S) // Wstride + 1
    
    # Pooling doesn't change channel count
    return None, (C, P, Q)  # Return updated shape only, no instance data

# Extract structural information from model with special handling for ResNet
def extract_structure(model, input_shape=(3, 224, 224), network_name=None):
    runtime_timer = runtime_tracker.start("extract_structure")
    
    # Special handling for ResNet models
    if network_name and network_name.lower().startswith("resnet"):
        return extract_resnet_structure(model, input_shape, network_name)
    
    # Regular structure extraction for other models
    shape = input_shape
    layer_structures = OrderedDict()
    layer_counter = 0

    for name, layer in model.named_modules():
        if name == "":
            continue  # skip top-level model

        if is_norm_layer(layer):
            continue

        # Add new shape calculation when pooling layers are encountered
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            _, shape = pool_instance(layer, shape)
            # We don't create a layer entry for pooling, just update the shape
            continue
        elif isinstance(layer, nn.Conv2d):
            inst, shape = conv2d_instance(layer, shape)
            layer_type = "Conv2d"
        elif isinstance(layer, nn.Linear):
            inst, shape = linear_instance(layer, shape)
            layer_type = "Linear"
        else:
            continue

        # We need to track layer by its position (for matching with histograms later)
        layer_id = f"{layer_counter:02d}"
        layer_counter += 1
        
        # filter only non-default values
        clean_inst = {k: v for k, v in inst.items() if str(v) != str(DEFAULTS.get(k))}
        layer_structures[layer_id] = {
            "name": name,
            "type": layer_type,
            "instance": clean_inst
        }
    
    runtime_tracker.stop(runtime_timer)
    return layer_structures

# Helper to extract ResNet structure correctly
def extract_resnet_structure(model, input_shape=(3, 224, 224), network_name=None):
    """Special extraction for ResNet that handles shortcut connections properly"""
    print(f"Using specialized ResNet structure extraction for {network_name}")
    
    # Initial shape after input
    shape = input_shape
    layer_structures = OrderedDict()
    layer_counter = 0
    
    # First, identify all convolution and linear layers
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            conv_layers.append((name, module))
    
    # Initial convolution (layer 0)
    name, layer = conv_layers[0]
    inst, shape = conv2d_instance(layer, shape)
    layer_id = f"{layer_counter:02d}"
    layer_counter += 1
    clean_inst = {k: v for k, v in inst.items() if str(v) != str(DEFAULTS.get(k))}
    layer_structures[layer_id] = {
        "name": name, "type": "Conv2d", "instance": clean_inst
    }
    
    # First maxpool in ResNet (not counted as a layer, just updates shape)
    # This is after the first conv layer
    maxpool_shape = shape
    maxpool_shape = (shape[0], shape[1] // 2, shape[2] // 2)  # Typical maxpool dims
    shape = maxpool_shape
    
    # Process remaining layers carefully to handle shortcut connections
    for i in range(1, len(conv_layers)):
        name, layer = conv_layers[i]
        
        # Check if this is a 1x1 conv (typically a shortcut)
        is_shortcut = False
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
            if kernel_size == (1, 1) and "downsample" in name:
                is_shortcut = True
        
        # Special handling for the final linear layer in ResNet
        if isinstance(layer, nn.Linear) and "fc" in name:
            # In ResNet, before the final FC layer, there's an adaptive pooling to 1x1
            # This means the input shape should just be the number of channels (not multiplied by spatial dims)
            if len(shape) > 1:
                # Just use the channel dimension as input to the FC layer
                channels = shape[0]
                inst = {
                    'C': channels,
                    'M': layer.out_features
                }
                shape = (layer.out_features,)
            else:
                # Normal linear layer processing if already flattened
                inst, shape = linear_instance(layer, shape)
        # If we're at a boundary where spatial dims change (layers 5, 9, etc in ResNet18)
        # These are the first layers of new blocks that reduce spatial dimensions
        elif "layer" in name and any(x in name for x in ["0.conv1", "0.downsample.0"]):
            # For the main path, we'll have stride=2
            if not is_shortcut and isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                # This is a main path with stride=2, leading to output dim / 2
                inst, shape = conv2d_instance(layer, shape)
            elif is_shortcut:
                # For shortcut with stride=2, calc dims separately
                shortcut_input_shape = shape
                # Note: shortcut has its own input shape!
                inst, _ = conv2d_instance(layer, shortcut_input_shape)
            else:
                # Regular conv in a new block without dim change
                inst, shape = conv2d_instance(layer, shape)
        else:
            # Normal layer processing
            if isinstance(layer, nn.Conv2d):
                inst, shape = conv2d_instance(layer, shape)
            elif isinstance(layer, nn.Linear):
                inst, shape = linear_instance(layer, shape)
        
        # Store this layer
        layer_id = f"{layer_counter:02d}"
        layer_counter += 1
        clean_inst = {k: v for k, v in inst.items() if str(v) != str(DEFAULTS.get(k))}
        layer_structures[layer_id] = {
            "name": name, 
            "type": "Conv2d" if isinstance(layer, nn.Conv2d) else "Linear",
            "instance": clean_inst
        }
    
    return layer_structures

# Function to update histograms in chunks to avoid memory issues
def update_histogram(data, hist, num_bins=31):
    # Create fixed bin edges for consistency across all histograms
    min_val = -1.0
    max_val = 1.0
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    new_hist, _ = np.histogram(data, bins=bin_edges, density=False)
    return hist + new_hist, data.size, bin_edges

# Collect histogram data for model
def collect_histograms(model, device, dataset_name="cifar10", batch_size=64, num_bins=31):
    runtime_timer = runtime_tracker.start("collect_histograms")
    
    # Create common bin edges
    NUM_BINS = num_bins # Use the provided number of bins
    min_val = -1.0
    max_val = 1.0
    common_bin_edges = np.linspace(min_val, max_val, NUM_BINS + 1)
    
    # Dictionary to store all layer data
    layer_data = OrderedDict()
    hook_counter = 0
    
    # Get layers with weights
    prep_timer = runtime_tracker.start("prepare_hooks")
    layers_with_weights = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layers_with_weights.append((name, module))
    
    print(f"Found {len(layers_with_weights)} layers with weights")
    
    # Create data structures for each layer
    for layer_name, layer in layers_with_weights:
        # Initialize histograms for this layer
        layer_data[layer_name] = {
            'input_hist': np.zeros(NUM_BINS),
            'weight_hist': np.zeros(NUM_BINS),
            'output_hist': np.zeros(NUM_BINS),
            'input_samples': 0,
            'output_samples': 0,
            'hook_in_calls': 0,
            'hook_out_calls': 0,
            'layer_type': layer.__class__.__name__,
            'bin_edges': common_bin_edges,
        }
        
        # Extract weights for this layer
        weights = layer.weight.data.cpu().numpy().flatten()
        
        # Compute weight histogram
        layer_data[layer_name]['weight_hist'], _, _ = update_histogram(weights, layer_data[layer_name]['weight_hist'], num_bins=NUM_BINS)
        
        print(f"Processed weights for layer: {layer_name} ({layer.__class__.__name__})")
    runtime_tracker.stop(prep_timer)
    
    # Register hooks for each layer
    def make_hook_fn(layer_name, is_input=True):
        def hook_fn(module, inp, out):
            nonlocal hook_counter
            hook_counter += 1
            
            # Extract appropriate tensor based on hook type
            if is_input:
                # For input hooks, extract the input tensor
                layer_data[layer_name]['hook_in_calls'] += 1
                data = inp[0].detach().cpu().numpy().flatten()
                hist_key = 'input_hist'
                samples_key = 'input_samples'
            else:
                # For output hooks, extract the output tensor
                layer_data[layer_name]['hook_out_calls'] += 1
                data = out.detach().cpu().numpy().flatten()
                hist_key = 'output_hist'
                samples_key = 'output_samples'
            
            # Update histogram
            layer_data[layer_name][hist_key], samples, _ = update_histogram(
                data, layer_data[layer_name][hist_key], num_bins=NUM_BINS)
            layer_data[layer_name][samples_key] += samples
            
            # Print status occasionally
            if hook_counter % 100 == 0:
                print(f"\nProcessed {hook_counter} hook calls across all layers")
            
            # Clean up
            del data
        
        return hook_fn
    
    # Register hooks for all layers
    hook_timer = runtime_tracker.start("register_hooks")
    hook_handles = []
    for layer_name, layer in layers_with_weights:
        # Register input hook
        handle_in = layer.register_forward_pre_hook(lambda module, inp, name=layer_name: 
                                                  make_hook_fn(name, is_input=True)(module, inp, None))
        hook_handles.append(handle_in)
        
        # Register output hook
        handle_out = layer.register_forward_hook(make_hook_fn(layer_name, is_input=False))
        hook_handles.append(handle_out)
    runtime_tracker.stop(hook_timer)
    
    # Define transformations
    data_prep_timer = runtime_tracker.start("prepare_dataset")
    
    # Define common resize transform
    resize_transform = transforms.Resize(224) # Many models expect 224x224
    
    # Dataset-specific normalization
    if dataset_name.lower() == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        # Note: ImageNet often uses CenterCrop(224) after Resize(256), 
        # but Resize(224) is common too and simpler here.
        transform = transforms.Compose([
            resize_transform, 
            transforms.ToTensor(),
            normalize
        ])
        dataset_root = './data/imagenet' # Default path, might need adjustment
        # Ensure the user has the ImageNet dataset downloaded here.
        # We use the 'train' split for generating histograms as it's larger.
        # Using 'val' might be faster for testing if available.
        try:
            dataset = datasets.ImageNet(root=dataset_root, split='train', transform=transform)
            print(f"Loading ImageNet dataset from {dataset_root} (train split)")
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Error loading ImageNet dataset from '{dataset_root}'.")
            print(f"Please ensure ImageNet is downloaded and extracted to that location.")
            print(f"Error details: {e}")
            sys.exit(1) # Exit if dataset cannot be loaded
            
    elif dataset_name.lower() == "cifar100":
        # CIFAR-100 uses the same normalization as CIFAR-10 typically
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], # Specific CIFAR100 stats
                                         std=[0.2675, 0.2565, 0.2761])
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            normalize
        ])
        dataset_root = './data'
        print("Loading CIFAR-100 dataset...")
        dataset = datasets.CIFAR100(root=dataset_root, train=False, # Using test set for consistency
                                     download=True, transform=transform)
                                     
    elif dataset_name.lower() == "cifar10":
        # Default CIFAR-10 normalization
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], # Specific CIFAR10 stats
                                         std=[0.2023, 0.1994, 0.2010])
        transform = transforms.Compose([
            resize_transform,
            transforms.ToTensor(),
            normalize
        ])
        dataset_root = './data'
        print("Loading CIFAR-10 dataset...")
        dataset = datasets.CIFAR10(root=dataset_root, train=False, # Using test set
                                     download=True, transform=transform)
    else:
        print(f"Error: Unsupported dataset name '{dataset_name}'. Please use 'cifar10', 'cifar100', or 'imagenet'.")
        sys.exit(1)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                             pin_memory=True, num_workers=2)
    print(f"Dataset size: {len(dataset)} images")
    runtime_tracker.stop(data_prep_timer)
    
    # Run inference with progress tracking
    print("Starting inference...")
    inference_timer = runtime_tracker.start("inference")
    model.eval()
    start_time = time.time()
    last_update_time = start_time
    
    with torch.no_grad():
        # Count total batches for progress tracking
        total_batches = len(data_loader)
        print(f"Processing {total_batches} batches...")
        
        # Track GPU memory before starting
        if torch.cuda.is_available():
            print(f"GPU memory before inference: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        if has_tqdm:
            # Use tqdm for a nice progress bar
            for batch in tqdm(data_loader, desc="Running inference"):
                # Move input data to GPU
                inputs_gpu = batch[0].to(device, non_blocking=True)
                output = model(inputs_gpu)
                # Force CUDA synchronization to ensure GPU computation is complete
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        else:
            # Simple percentage based tracking
            for i, batch in enumerate(data_loader):
                # Move input data to GPU
                inputs_gpu = batch[0].to(device, non_blocking=True)
                output = model(inputs_gpu)
                # Force CUDA synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Report progress periodically
                current_time = time.time()
                if (i + 1) % max(1, total_batches // 10) == 0 or (i + 1) == total_batches or current_time - last_update_time > 5:
                    if torch.cuda.is_available():
                        cur_mem = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"Progress: {(i + 1) / total_batches:.1%} ({i + 1}/{total_batches}) | GPU memory: {cur_mem:.2f} GB")
                    else:
                        print(f"Progress: {(i + 1) / total_batches:.1%} ({i + 1}/{total_batches})")
                    last_update_time = current_time
    
    # Remove all hooks
    for handle in hook_handles:
        handle.remove()
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    runtime_tracker.stop(inference_timer)
    
    # Function to normalize and format histogram data
    def normalize_and_format_hist(hist, num_samples):
        if num_samples > 0:
            normalized_hist = hist / num_samples
        else:
            normalized_hist = hist
        
        # Return as a list of float values
        return [float(val) for val in normalized_hist]
    
    # Format histogram data
    post_process_timer = runtime_tracker.start("post_process_histograms")
    layer_histograms = OrderedDict()
    for layer_name, data in layer_data.items():
        input_hist = normalize_and_format_hist(data['input_hist'], data['input_samples'])
        
        # Fix for weight histogram - normalize by sum to create a distribution that sums to 1.0
        weight_hist_sum = np.sum(data['weight_hist'])
        if weight_hist_sum > 0:
            weight_hist = [float(val/weight_hist_sum) for val in data['weight_hist']]
        else:
            weight_hist = [float(val) for val in data['weight_hist']]
            
        output_hist = normalize_and_format_hist(data['output_hist'], data['output_samples'])
        
        layer_histograms[layer_name] = {
            'type': data['layer_type'],
            'histograms': {
                'Inputs': input_hist,
                'Weights': weight_hist,
                'Outputs': output_hist
            }
        }
    runtime_tracker.stop(post_process_timer)
    
    runtime_tracker.stop(runtime_timer)
    return layer_histograms

# Combine structure and histogram data
def combine_data(structure_data, histogram_data):
    runtime_timer = runtime_tracker.start("combine_data")
    combined_data = OrderedDict()
    
    # Map histogram data to structure data using layer names
    histogram_by_name = {name: data for name, data in histogram_data.items()}
    
    # Create combined data for each layer
    layer_index = 0
    for layer_id, struct in structure_data.items():
        layer_name = struct['name']
        
        # Find matching histogram data
        if layer_name in histogram_by_name:
            hist_data = histogram_by_name[layer_name]['histograms']
            layer_type = struct['type']
            
            # Create combined layer data
            combined_data[layer_id] = {
                'name': layer_name,
                'type': layer_type,
                'instance': struct['instance'],
                'histograms': hist_data
            }
            
            layer_index += 1
    
    runtime_tracker.stop(runtime_timer)
    return combined_data

# Generate YAML content for a layer
def generate_yaml_content(layer_data, dnn_name):
    yaml_content = """{{include_text('../problem_base.yaml')}}
problem:
  <<<: *problem_base
"""
    
    # Fix capitalization for stride parameters to match expected format
    instance_data = layer_data['instance'].copy()
    if 'Hstride' in instance_data:
        instance_data['HStride'] = instance_data.pop('Hstride')
    if 'Wstride' in instance_data:
        instance_data['WStride'] = instance_data.pop('Wstride')
    if 'Hdilation' in instance_data:
        instance_data['HDilation'] = instance_data.pop('Hdilation')
    if 'Wdilation' in instance_data:
        instance_data['WDilation'] = instance_data.pop('Wdilation')
    
    # Format instance data as a single line with curly braces
    instance_parts = [f"{k}: {v}" for k, v in instance_data.items()]
    instance_str = "{" + ", ".join(instance_parts) + "}"
    yaml_content += f"  instance: {instance_str}\n\n"
    
    # Add metadata
    yaml_content += f"  name: {layer_data['type']}\n"
    yaml_content += f"  dnn_name: {dnn_name}\n"
    yaml_content += f"  notes: {layer_data['type']}\n"
    
    # Add histogram information
    yaml_content += """  # These histograms symmetric and zero-centered (the centermost bin is the
  # probability of zero). Histograms are normalized to sum to 1.0 and they have
  # 2^N-1 bins for some integer N. Higher N yields higher-fidelity histograms,
  # but also increases runtime & the size of YAML files. Encoding functions will
  # upsample or downsample histograms depending on the bitwidth of the
  # corresponding operands.
  histograms:
"""
    
    # Add histogram data
    for hist_name, hist_data in layer_data['histograms'].items():
        formatted_hist = [f"{val:.6g}" for val in hist_data]
        yaml_content += f"    {hist_name}: [{','.join(formatted_hist)}]\n"
    
    return yaml_content

# Main function to generate network definition
def generate_network_definition(network_name, dataset_name="cifar10", batch_size=64, force=False, num_bins=31):
    total_timer = runtime_tracker.start("total")
    
    # Set device
    if not torch.cuda.is_available():
        print("CUDA is not available! Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        try:
            # Use newer CUDA memory API if available (PyTorch 1.10+)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            
            # memory_reserved is not reliable in newer PyTorch versions, check if available
            if hasattr(torch.cuda, 'memory_reserved'):
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
                print(f"Using GPU: {gpu_name}")
                print(f"GPU Memory: Total {gpu_mem_total:.2f} GB | Reserved: {gpu_mem_reserved:.2f} GB | Allocated: {gpu_mem_allocated:.2f} GB")
            else:
                print(f"Using GPU: {gpu_name}")
                print(f"GPU Memory: Total {gpu_mem_total:.2f} GB | Allocated: {gpu_mem_allocated:.2f} GB")
        except (RuntimeError, AttributeError) as e:
            print(f"Using GPU: {gpu_name} (Memory info unavailable: {str(e)})")
        
        # Clear any cached memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Check if output directory exists
    output_dir = f"assets/DNN_models/{network_name}"
    if os.path.exists(output_dir) and not force:
        print(f"Warning: Directory '{output_dir}' already exists!")
        response = input("Do you want to continue and overwrite existing files? (y/n): ")
        if response.lower() != 'y':
            print("Operation aborted.")
            return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading {network_name} model...")
    model_timer = runtime_tracker.start("load_model")
    try:
        # Try to load from torchvision.models
        model_fn = getattr(models, network_name.lower())
        
        # Modern approach using weights parameter
        try:
            # Import weights directly from torchvision.models
            weights_module = __import__('torchvision.models', fromlist=['*'])
            
            # Try standard weight class naming patterns
            weight_class_names = [
                f"{network_name}_Weights",  # Standard format
                f"{network_name.upper()}_Weights",  # For models like VGG16
                f"{network_name.capitalize()}_Weights",  # For simple names
                # Handle specific model families
                f"ResNet{network_name[6:]}_Weights" if network_name.lower().startswith("resnet") else None,
                f"VGG{network_name[3:]}_Weights" if network_name.lower().startswith("vgg") else None,
                f"DenseNet{network_name[8:]}_Weights" if network_name.lower().startswith("densenet") else None,
            ]
            
            # Try each possible weights class name
            weights_class = None
            for class_name in weight_class_names:
                if class_name is None:
                    continue
                try:
                    weights_class = getattr(weights_module, class_name)
                    print(f"Found weights class: {class_name}")
                    break
                except AttributeError:
                    continue
            
            if weights_class is not None:
                print(f"Loading {network_name} with {weights_class.DEFAULT} weights")
                model = model_fn(weights=weights_class.DEFAULT)
            else:
                # Fallback to no pre-trained weights
                print(f"No matching weights class found. Loading model without pre-trained weights.")
                model = model_fn(weights=None)
                
        except (AttributeError, ImportError) as e:
            print(f"Warning: Error loading weights for {network_name}: {str(e)}")
            print("Loading model without pre-trained weights")
            model = model_fn(weights=None)  # Modern API with no pre-trained weights
        
    except (AttributeError, ValueError) as e:
        print(f"Model '{network_name}' not found in torchvision.models: {str(e)}")
        runtime_tracker.stop(model_timer)
        runtime_tracker.stop(total_timer)
        return
    
    model = model.to(device)
    print(f"Model is on CUDA: {next(model.parameters()).is_cuda if torch.cuda.is_available() else False}")
    runtime_tracker.stop(model_timer)
    
    # Extract structure with network-specific handling
    print("Extracting model structure...")
    structure_data = extract_structure(model, network_name=network_name)
    
    # Collect histograms
    print("Collecting histogram data...")
    histogram_data = collect_histograms(model, device, dataset_name, batch_size, num_bins)
    
    # Combine data
    print("Combining structure and histogram data...")
    combined_data = combine_data(structure_data, histogram_data)
    
    # Generate and save YAML files
    print(f"Generating YAML files in {output_dir}...")
    yaml_timer = runtime_tracker.start("generate_yaml_files")
    
    # Determine the number of digits needed for zero-padding
    num_layers = len(combined_data)
    num_digits = len(str(num_layers - 1)) if num_layers > 0 else 1
    
    # Use enumerate to get the index for filename formatting
    for i, (layer_id, layer_data) in enumerate(combined_data.items()):
        yaml_content = generate_yaml_content(layer_data, network_name)
        
        # Format filename based on index and required digits
        filename = f"{i:0{num_digits}d}.yaml"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"Created {file_path}")
    runtime_tracker.stop(yaml_timer)
    
    print(f"Successfully generated network definition for {network_name} with {len(combined_data)} layers.")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print runtime summary
    runtime_tracker.stop(total_timer)
    runtime_tracker.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate network definition files for neural network models")
    parser.add_argument("--network-name", "-n", type=str, default="resnet18", 
                        help="Name of the network (e.g. resnet18, vgg16, alexnet)")
    parser.add_argument("--dataset", "-d", type=str, default="cifar10", 
                        help="Dataset name (e.g. cifar10, cifar100, imagenet)")
    parser.add_argument("--batch-size", "-b", type=int, default=64, 
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Force overwrite if output directory exists")
    parser.add_argument("--num-bins", "-nb", type=int, default=31,
                        help="Number of bins for histograms (default: 31)")
    
    args = parser.parse_args()
    
    generate_network_definition(args.network_name, args.dataset, args.batch_size, args.force, args.num_bins) 