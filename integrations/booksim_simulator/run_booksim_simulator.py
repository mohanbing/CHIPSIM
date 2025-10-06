import numpy as np
import os
import shutil
import time
import argparse
import sys

def create_traffic_trace(transition_mtx):
    # Set the number of chiplets in the system
    num_chiplets = np.shape(transition_mtx)[0]

    # Create the scale matrix to find the scale for each index in the transition matrix
    scale_mtx = np.full_like(transition_mtx, 1)

    # Populate the scale transition_mtx
    # For every row
    for n in range(0,num_chiplets):
        # For every col
        for m in range(0,num_chiplets):
            # Scale equals lowest value in the transition matrix
            # If the transition_mtxay at the location has a value, calculate the scale factor
            if (transition_mtx[n][m] != 0):
                count = 0    # Track the amount of decimal places
                temp = transition_mtx[n][m]  # Save a temp variable to manipulate without changing the transition_mtx
                while(temp != 0):   # Until temp = 0
                    temp //= 10     # Divide temp by 10
                    count += 1      # Inc count
                # Ensure the count is not 0, if it is the scale_mtx at (index) stays 1
                if(count > 0):
                    scale_mtx[n][m] = 10**(count-1)

    # Create transition_mtxay of scaling factors for each row of the matrix
    scale_arr = np.ones(num_chiplets)

    # Loop for number of rows in scale_mtx
    for o in range(num_chiplets):
        if (max(scale_mtx[o]) > 10000):
            scale_arr[o] = 10000
            transition_mtx[o] = np.ceil(transition_mtx[o] / 10000) # Scale down the transition_mtx
        elif (max(scale_mtx[o]) > 100):
            scale_arr[o] = 100
            transition_mtx[o] = np.ceil(transition_mtx[o] / 100)   # Scale down the transition_mtx

    # Create a single combined trace array for all chiplets
    combined_trace = []
    
    # For each x index (source chiplet)
    for x in range(0, num_chiplets):
        # For each y index (destination chiplet)
        for y in range(0, num_chiplets):
            # Add traces for this source-destination pair
            k = 0
            while(transition_mtx[x][y] > 0):
                transition_mtx[x][y] -= 1
                trace = (x, y, k)
                combined_trace.append(trace)
                k += 1

    # Sort the combined trace by injection time (k, which is the third element in each tuple)
    # Then sort subgroups with identical k values by source chiplet (the first element in each tuple)
    # Sort trace[2] ascending and trace[0] descending (high to low)
    combined_trace.sort(key=lambda trace: (trace[2], -trace[0]))

    return scale_arr, combined_trace

def create_anynet_folder(anynet_template_dir, working_dir, architecture, num_chiplets):
    # Create the anynet folder in the single_matrix_simulator directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(script_dir, 'temp_AnyNet_sim')
    
    # Remove existing folder if it exists
    if os.path.exists(dest):
        shutil.rmtree(dest)
    
    # Copy the template AnyNet folder
    shutil.copytree(anynet_template_dir, dest)
    
    # Copy the correct anynet_file for the architecture and number of chiplets
    source_file = os.path.join(dest, f'anynet_file_{architecture}_{num_chiplets}')
    dest_file = os.path.join(dest, 'anynet_file')
    
    if os.path.exists(source_file):
        shutil.copy(source_file, dest_file)
    else:
        print(f"Error: anynet_file_{architecture}_{num_chiplets} does not exist in template directory")
        return False, dest
    
    return True, dest

def run_booksim_simulation(transition_matrix, anynet_template_dir, working_dir, architecture, num_chiplets):
    # Create anynet folder
    success, anynet_folder = create_anynet_folder(anynet_template_dir, working_dir, architecture, num_chiplets)
    if not success:
        return None, None
    
    # Store current directory
    original_dir = os.getcwd()
    
    # Create debug directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(script_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    try:
        # Generate traffic traces from the transition matrix
        scaling_factors, combined_traces = create_traffic_trace(transition_matrix)
        
        print(f"Running combined simulation for all chiplets")
        
        # Open the tracefile which Booksim will use
        trace_file_path = os.path.join(anynet_folder, 'trace_file.txt')
        with open(trace_file_path, 'w') as file:
            # Write the combined traces to the file
            for line in combined_traces:
                # Replace extraneous characters in the input
                line = str(line)
                line = line.replace(',', '')
                line = line.replace('(', '')
                line = line.replace(')', '')
                line = line + '\n'
                file.write(line)
        
        # Save a debug copy of the trace file
        timestamp = time.strftime("%Y.%m.%d_%H.%M.%S")
        debug_trace_path = os.path.join(debug_dir, f'trace_file_{architecture}_{num_chiplets}_{timestamp}.txt')
        shutil.copy2(trace_file_path, debug_trace_path)
        print(f"Saved debug trace file to: {debug_trace_path}")
        
        # Change directory to the AnyNet_sim folder
        os.chdir(anynet_folder)
        
        # Call BookSim and redirect output to cat.txt
        booksim_cmd = './booksim anynet_config > cat.txt'
        os.system(booksim_cmd)
        
        # Get the raw booksim latency
        latency = os.popen('grep "Trace is finished in" cat.txt | tail -1 | awk \'{print $5}\'').read().strip()
        
        # If latency could not be extracted, raise an error and end the program
        if not latency:
            print("❌ ERROR: Could not extract latency from simulation")
            print("   This indicates a critical failure in the BookSim simulation.")
            print("   Please check the simulation configuration and trace files.")
            sys.exit(1)  # Exit the program with error code
            
        # Convert to float and multiply by technology parameter
        system_latency = float(latency) * 0.8695

        # Get power information
        power = os.popen('grep "Total Power" cat.txt | tail -1 | awk \'{print $4}\'').read().strip()
        # If power could not be extracted, set to 0
        if not power:
            print("Warning: Could not extract power from simulation")
            power = "0"
            
        # Calculate total energy (latency * power)
        total_energy = system_latency * float(power)
        
        # Also copy the output file for debugging
        debug_output_path = os.path.join(debug_dir, f'booksim_output_{architecture}_{num_chiplets}_{timestamp}.txt')
        shutil.copy2(os.path.join(anynet_folder, 'cat.txt'), debug_output_path)
        print(f"Saved BookSim output to: {debug_output_path}")
        
        # Return to original directory
        os.chdir(original_dir)
        
        return system_latency, total_energy
    
    finally:
        # Return to original directory if not already there
        if os.getcwd() != original_dir:
            os.chdir(original_dir)
            
        # Clean up the temporary folder but no need to delete trace files as we've saved copies
        # if os.path.exists(anynet_folder):
        #     shutil.rmtree(anynet_folder)
        #     print(f"Cleaned up temporary folder: {anynet_folder}")
        #     print(f"Debug files preserved in: {debug_dir}")

def print_results(system_latency, total_energy):
    """
    Print simulation results to the terminal.
    """
    print("\nSimulation Results:")
    print("-" * 40)
    
    # Print the system latency
    print(f"System Latency: {system_latency:.2f}")
    print(f"Total Energy: {total_energy:.2f}")
    
    return system_latency, total_energy

def simulate_matrix(transition_matrix, anynet_template_dir, working_dir=os.getcwd(), 
                   architecture='Floret', num_chiplets=None,
                   verbose=True):
    """
    Simulate a transition matrix using BookSim.
    
    Parameters:
    -----------
    transition_matrix : numpy.ndarray or str
        The transition matrix as a NumPy array or path to a CSV file
    anynet_template_dir : str
        Path to the AnyNet template directory
    working_dir : str, optional
        Working directory for simulation (default: current directory)
    architecture : str, optional
        Architecture to simulate (default: Floret)
    num_chiplets : int, optional
        Number of chiplets (default: inferred from matrix dimensions)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns:
    --------
    tuple
        (system_latency, total_energy) where:
        - system_latency: single latency value for the entire system
        - total_energy: total energy consumption
    """
    # Start timing
    start_time = time.time()
    
    # Load transition matrix if it's a file path
    if isinstance(transition_matrix, str):
        transition_matrix = np.loadtxt(transition_matrix, delimiter=',')
    
    # Determine number of chiplets if not provided
    if num_chiplets is None:
        num_chiplets = transition_matrix.shape[0]
    
    if verbose:
        print(f"Loaded transition matrix with shape {transition_matrix.shape}")
        print(f"Architecture: {architecture}, Chiplets: {num_chiplets}")
    
    # Run BookSim simulation
    system_latency, total_energy = run_booksim_simulation(
        transition_matrix, 
        anynet_template_dir,
        working_dir,
        architecture,
        num_chiplets
    )
    
    # Calculate and display elapsed time
    if verbose:
        elapsed_time = time.time() - start_time
        print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    
    return system_latency, total_energy

def main():
    # Load transition matrix from file
    transition_matrix = np.loadtxt(args.transition_matrix, delimiter=',')
    
    # Call the simulation function
    system_latency, total_energy = simulate_matrix(
        transition_matrix,
        args.anynet_template,
        args.working_dir,
        args.architecture,
        args.num_chiplets
    )
    
    # Return an error code if simulation failed
    if system_latency is None or total_energy is None:
        return 1
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate a single transition matrix using BookSim')
    
    parser.add_argument('--transition_matrix', required=True, 
                        help='Path to the transition matrix CSV file')
    parser.add_argument('--working_dir', default=os.getcwd(),
                        help='Working directory for simulation (default: current directory)')
    parser.add_argument('--architecture', default='Floret',
                        help='Architecture to simulate (default: Floret)')
    parser.add_argument('--num_chiplets', type=int, default=None,
                        help='Number of chiplets (default: inferred from matrix dimensions)')
    parser.add_argument('--anynet_template', required=True,
                        help='Path to the AnyNet template directory')
    
    args = parser.parse_args()
    exit(main()) 