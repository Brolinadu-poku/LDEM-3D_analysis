import os  # Import os module for operating system functionalities
import glob  # Import glob module for file pattern matching
import numpy as np  # Import numpy for numerical operations
import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from pathlib import Path  # Import Path class for handling file paths

# Set plotting style (using matplotlib built-in styles)
plt.style.use('default')  # Use default Matplotlib style
plt.rcParams['figure.facecolor'] = 'white'  # Set figure background color to white
plt.rcParams['axes.facecolor'] = 'white'  # Set axes background color to white
plt.rcParams['axes.grid'] = True  # Enable grid on axes
plt.rcParams['grid.alpha'] = 0.3  # Set grid transparency to 0.3
# Removed LaTeX rendering to avoid Unicode issues
# plt.rcParams['text.usetex'] = True
# plt.rcParams['mathtext.fontset'] = 'stix'

def read_coordination_analysis_file(filepath):
    """
    Utility function to read coordination analysis result files.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the coordination analysis result file
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'data': numpy array with all numerical data
        - 'metadata': dict with simulation parameters
        - 'columns': dict with column mappings
    """
    filepath = Path(filepath)  # Convert filepath to Path object
    
    # Read metadata from header
    metadata = {}  # Initialize empty dictionary for metadata
    with open(filepath, 'r') as f:  # Open file in read mode
        for line in f:  # Iterate through each line
            if not line.startswith('#'):  # Skip non-header lines
                break  # Exit loop if non-header line found
            if 'Volume Fraction' in line:  # Check for volume fraction line
                metadata['phi'] = float(line.split(':')[1].strip())  # Extract and store phi value
            elif 'Aspect Ratio' in line:  # Check for aspect ratio line
                metadata['aspect_ratio'] = float(line.split(':')[1].strip())  # Extract and store aspect ratio
            elif 'Velocity Ratio' in line:  # Check for velocity ratio line
                metadata['velocity_ratio'] = line.split(':')[1].strip()  # Extract and store velocity ratio
            elif 'Run Number' in line:  # Check for run number line
                metadata['run_number'] = int(line.split(':')[1].strip())  # Extract and store run number
            elif 'Number of Particles' in line:  # Check for number of particles line
                metadata['n_particles'] = int(line.split(':')[1].strip())  # Extract and store number of particles
            elif 'Box Dimensions' in line:  # Check for box dimensions line
                dims = line.split('-')[1].strip()  # Extract dimensions string
                for dim in dims.split(', '):  # Split dimensions
                    key, value = dim.split(': ')  # Split key-value pair
                    try:
                        metadata[key] = float(value)  # Store dimension as float
                    except ValueError:
                        metadata[key] = value  # Store as string if not float
    
    # Read numerical data
    data = np.loadtxt(filepath)  # Load numerical data from file into numpy array
    
    # Column mappings
    n_particles = metadata.get('n_particles', 1000)  # Get number of particles from metadata, default to 1000
    columns = {  # Define dictionary for column mappings
        'timestep_index': 0,  # Map timestep index to column 0
        'time': 1,  # Map time to column 1
        'shear_rate': 2,  # Map shear rate to column 2
        'viscosity': 3,  # Map viscosity to column 3
        'frictional_contact_number': 4  # Map frictional contact number to column 4
    }
    
    # Add coordination number columns
    for i in range(n_particles):  # Iterate through particles
        columns[f'coordination_number_particle_{i}'] = 5 + i  # Map coordination number columns starting from 5
    
    return {  # Return dictionary with results
        'data': data,  # Numerical data
        'metadata': metadata,  # Simulation metadata
        'columns': columns  # Column mappings
    }

def calculate_coordination_statistics(analysis_result):
    """
    Calculate Z(t), n(t), <Z>(t), and <n>(t) from coordination analysis data.
    Now using moving time averages and t=1 to T indexing.
    
    Parameters:
    -----------
    analysis_result : dict
        Result from read_coordination_analysis_file()
        
    Returns:
    --------
    dict : Dictionary containing calculated statistics
    """
    data = analysis_result['data']  # Extract numerical data from analysis result
    n_particles = analysis_result['metadata']['n_particles']  # Extract number of particles from metadata
    
    # Extract coordination numbers for all particles (columns 5 onwards)
    coord_numbers = data[:, 5:5+n_particles]  # Shape: (n_timesteps, n_particles)
    
    # Calculate Z(t) - average coordination number at each timestep (t=1 to T)
    Z_t = np.mean(coord_numbers, axis=1)  # Shape: (n_timesteps,)
    
    # Calculate n(t) - fraction of particles with frictional contacts at each timestep (t=1 to T)
    particles_with_contacts = coord_numbers > 0  # Boolean array for particles with contacts
    n_t = np.mean(particles_with_contacts, axis=1)  # Shape: (n_timesteps,)
    
    # Calculate moving time averages <Z>(t) and <n>(t)
    # For timestep t, average from timestep 1 to timestep t
    Z_moving_avg = np.zeros_like(Z_t)  # Initialize array for Z moving averages
    n_moving_avg = np.zeros_like(n_t)  # Initialize array for n moving averages
    
    for t in range(len(Z_t)):  # Iterate through timesteps
        # Moving average from timestep 1 to current timestep (t+1)
        Z_moving_avg[t] = np.mean(Z_t[:t+1])  # Average from index 0 to t
        n_moving_avg[t] = np.mean(n_t[:t+1])  # Average from index 0 to t
    
    # Final time averages (values at the last timestep)
    Z_final_avg = Z_moving_avg[-1]  # <Z> at final timestep T
    n_final_avg = n_moving_avg[-1]  # <n> at final timestep T
    
    # Extract other relevant data
    time = data[:, 1]  # Extract time column
    shear_rate = data[:, 2]  # Extract shear rate column
    viscosity = data[:, 3]  # Extract viscosity column
    
    return {  # Return dictionary with calculated statistics
        'Z_t': Z_t,  # Average coordination number at each timestep
        'n_t': n_t,  # Fraction of particles with frictional contacts at each timestep
        'Z_moving_avg': Z_moving_avg,  # Moving time-averaged coordination number <Z>(t)
        'n_moving_avg': n_moving_avg,  # Moving time-averaged fraction of particles with frictional contacts <n>(t)
        'Z_final_avg': Z_final_avg,  # Final time-averaged coordination number <Z>
        'n_final_avg': n_final_avg,  # Final time-averaged fraction of particles with frictional contacts <n>
        'time': time,  # Time array (t=1 to T)
        'shear_rate': shear_rate,  # Shear rate array (t=1 to T)
        'viscosity': viscosity,  # Viscosity array (t=1 to T)
        'coord_numbers': coord_numbers  # Full coordination number matrix
    }

def process_all_phi_values(data_directory):
    """
    Process coordination analysis files for all phi values.
    
    Parameters:
    -----------
    data_directory : str or Path
        Directory containing coordination analysis files
        
    Returns:
    --------
    dict : Dictionary with phi values as keys and statistics as values
    """
    data_directory = Path(data_directory)  # Convert data_directory to Path object
    phi_values = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]  # Define list of phi values
    
    results = {}  # Initialize empty dictionary for results
    
    for phi in phi_values:  # Iterate through phi values
        phi_str = f"{phi:.2f}"  # Format phi as two-decimal string
        print(f"Processing phi = {phi} ({phi_str})...")  # Print processing status
        
        # Find coordination analysis files for this phi value
        # Search recursively in case files are in subdirectories
        pattern = f"coordination_analysis_phi_{phi_str}_*.txt"  # Define file pattern
        files = list(data_directory.glob(pattern))  # Direct search
        
        # If no files found in main directory, search recursively
        if not files:  # Check if files list is empty
            files = list(data_directory.rglob(pattern))  # Recursive search
        
        # Also try searching with different phi formatting (e.g., 0.520 vs 0.52)
        if not files:  # Check if files list is still empty
            phi_str_alt = f"{phi:.3f}"  # Format phi as three-decimal string
            pattern_alt = f"coordination_analysis_phi_{phi_str_alt}_*.txt"  # Define alternative pattern
            files = list(data_directory.glob(pattern_alt))  # Direct search with alternative pattern
            if not files:  # Check if files list is empty
                files = list(data_directory.rglob(pattern_alt))  # Recursive search with alternative pattern
        
        if not files:  # Check if no files found
            print(f"  No files found for phi = {phi} (searched patterns: {pattern} and coordination_analysis_phi_{phi:.3f}_*.txt)")  # Print no files message
            print(f"  Searched in: {data_directory}")  # Print search directory
            continue  # Skip to next phi value
        
        phi_results = []  # Initialize list for results of this phi
        
        for file_path in files:  # Iterate through found files
            try:
                # Read and analyze file
                analysis_result = read_coordination_analysis_file(file_path)  # Read file
                stats = calculate_coordination_statistics(analysis_result)  # Calculate statistics
                
                # Add metadata
                stats['metadata'] = analysis_result['metadata']  # Add metadata to stats
                stats['filename'] = file_path.name  # Add filename to stats
                
                phi_results.append(stats)  # Append stats to phi results
                print(f"  Processed: {file_path.name}")  # Print processed file message
                
            except Exception as e:  # Catch any exceptions
                print(f"  Error processing {file_path.name}: {e}")  # Print error message
                continue  # Skip to next file
        
        if phi_results:  # Check if phi results are not empty
            results[phi] = phi_results  # Store phi results in main results dictionary
            
            # Print summary statistics for this phi
            all_Z_final_avg = [result['Z_final_avg'] for result in phi_results]  # Collect final Z averages
            all_n_final_avg = [result['n_final_avg'] for result in phi_results]  # Collect final n averages
            
            print(f"  Summary for phi = {phi}:")  # Print summary header
            print(f"    <Z> (final): {np.min(all_Z_final_avg):.3f} - {np.max(all_Z_final_avg):.3f}")  # Print Z range
            print(f"    <n> (final): {np.min(all_n_final_avg):.3f} - {np.max(all_n_final_avg):.3f}")  # Print n range
            print(f"    Number of files: {len(phi_results)}")  # Print number of files processed
    
    return results  # Return main results dictionary

def create_plots(results, output_directory):
    """
    Create plots for viscosity, Z(t), n(t), and moving averages vs shear rate.
    
    Parameters:
    -----------
    results : dict
        Results from process_all_phi_values()
    output_directory : str or Path
        Directory to save plots
    """
    output_directory = Path(output_directory)  # Convert output_directory to Path object
    output_directory.mkdir(exist_ok=True)  # Create output directory if not existing
    
    # Colors for different phi values (using matplotlib colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # Generate colors from tab10 colormap
    
    # Create individual plots for each phi value
    for i, (phi, phi_results) in enumerate(results.items()):  # Iterate through results
        color = colors[i]  # Assign color for this phi
        
        # Create subplots for this phi value (5 plots in a single row)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # Create 1x5 subplot grid
        fig.suptitle(f'Coordination Analysis for φ = {phi}', fontsize=16)  # Set figure title
        
        for j, result in enumerate(phi_results):  # Iterate through results for this phi
            # Plot 1: Viscosity vs Shear Rate
            axes[0].plot(result['shear_rate'], result['viscosity'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)  # Plot viscosity
            axes[0].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
            axes[0].set_ylabel('Viscosity')  # Set y-label
            axes[0].set_title('Viscosity vs Shear Rate')  # Set plot title
            axes[0].legend()  # Add legend
            axes[0].grid(True, alpha=0.3)  # Enable grid
            
            # Plot 2: Z(t) vs Shear Rate
            axes[1].plot(result['shear_rate'], result['Z_t'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)  # Plot Z(t)
            axes[1].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
            axes[1].set_ylabel('Z(t)')  # Set y-label
            axes[1].set_title('Z(t) vs Shear Rate')  # Set plot title
            axes[1].legend()  # Add legend
            axes[1].grid(True, alpha=0.3)  # Enable grid
            
            # Plot 3: n(t) vs Shear Rate
            axes[2].plot(result['shear_rate'], result['n_t'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)  # Plot n(t)
            axes[2].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
            axes[2].set_ylabel('n(t) (Fraction of Particles in Frictional Contact)')  # Set y-label
            axes[2].set_title('n(t) vs Shear Rate')  # Set plot title
            axes[2].legend()  # Add legend
            axes[2].grid(True, alpha=0.3)  # Enable grid
            
            # Plot 4: <Z>(t) vs Shear Rate
            axes[3].plot(result['shear_rate'], result['Z_moving_avg'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)  # Plot <Z>(t)
            axes[3].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
            axes[3].set_ylabel('<Z>(t)')  # Set y-label
            axes[3].set_title('<Z>(t) vs Shear Rate')  # Set plot title
            axes[3].legend()  # Add legend
            axes[3].grid(True, alpha=0.3)  # Enable grid
            
            # Plot 5: <n>(t) vs Shear Rate
            axes[4].plot(result['shear_rate'], result['n_moving_avg'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)  # Plot <n>(t)
            axes[4].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
            axes[4].set_ylabel('<n>(t) (Fraction of Particles in Frictional Contact)')  # Set y-label
            axes[4].set_title('<n>(t) vs Shear Rate')  # Set plot title
            axes[4].legend()  # Add legend
            axes[4].grid(True, alpha=0.3)  # Enable grid
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(output_directory / f'coordination_analysis_phi_{phi:.2f}.png', 
                    dpi=600, bbox_inches='tight')  # Save plot with high DPI
        plt.close()  # Close plot
    
    # Create combined plots showing all phi values
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))  # Create 1x5 subplot grid
    fig.suptitle('Coordination Analysis - All φ Values', fontsize=16)  # Set figure title
    
    for i, (phi, phi_results) in enumerate(results.items()):  # Iterate through results
        color = colors[i % len(colors)]  # Assign color for this phi
        
        # Average across all runs for this phi (if multiple runs exist)
        all_shear_rates = []  # Initialize list for shear rates
        all_viscosities = []  # Initialize list for viscosities
        all_Z_t = []  # Initialize list for Z(t)
        all_n_t = []  # Initialize list for n(t)
        all_Z_moving_avg = []  # Initialize list for <Z>(t)
        all_n_moving_avg = []  # Initialize list for <n>(t)
        
        for result in phi_results:  # Iterate through results for this phi
            all_shear_rates.append(result['shear_rate'])  # Append shear rate
            all_viscosities.append(result['viscosity'])  # Append viscosity
            all_Z_t.append(result['Z_t'])  # Append Z(t)
            all_n_t.append(result['n_t'])  # Append n(t)
            all_Z_moving_avg.append(result['Z_moving_avg'])  # Append <Z>(t)
            all_n_moving_avg.append(result['n_moving_avg'])  # Append <n>(t)
        
        # If multiple runs, average them (assuming same timesteps)
        if len(phi_results) > 1:  # Check if multiple runs
            shear_rate = np.mean(all_shear_rates, axis=0)  # Average shear rates
            viscosity = np.mean(all_viscosities, axis=0)  # Average viscosities
            Z_t = np.mean(all_Z_t, axis=0)  # Average Z(t)
            n_t = np.mean(all_n_t, axis=0)  # Average n(t)
            Z_moving_avg = np.mean(all_Z_moving_avg, axis=0)  # Average <Z>(t)
            n_moving_avg = np.mean(all_n_moving_avg, axis=0)  # Average <n>(t)
        else:  # Single run
            shear_rate = all_shear_rates[0]  # Use single shear rate
            viscosity = all_viscosities[0]  # Use single viscosity
            Z_t = all_Z_t[0]  # Use single Z(t)
            n_t = all_n_t[0]  # Use single n(t)
            Z_moving_avg = all_Z_moving_avg[0]  # Use single <Z>(t)
            n_moving_avg = all_n_moving_avg[0]  # Use single <n>(t)
        
        # Plot combined results
        axes[0].plot(shear_rate, viscosity, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='o', markersize=2)  # Plot viscosity vs shear rate
        axes[1].plot(shear_rate, Z_t, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='s', markersize=2)  # Plot Z(t) vs shear rate
        axes[2].plot(shear_rate, n_t, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='^', markersize=2)  # Plot n(t) vs shear rate
        axes[3].plot(shear_rate, Z_moving_avg, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='d', markersize=2)  # Plot <Z>(t) vs shear rate
        axes[4].plot(shear_rate, n_moving_avg, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='v', markersize=2)  # Plot <n>(t) vs shear rate
    
    # Set labels and formatting for combined plots
    axes[0].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
    axes[0].set_ylabel('Viscosity')  # Set y-label
    axes[0].set_title('Viscosity vs Shear Rate - All φ Values')  # Set plot title
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend
    axes[0].grid(True, alpha=0.3)  # Enable grid
    
    axes[1].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
    axes[1].set_ylabel('Z(t)')  # Set y-label
    axes[1].set_title('Z(t) vs Shear Rate - All φ Values')  # Set plot title
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend
    axes[1].grid(True, alpha=0.3)  # Enable grid
    
    axes[2].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
    axes[2].set_ylabel('n(t) (Fraction of Particles in Frictional Contact)')  # Set y-label
    axes[2].set_title('n(t) vs Shear Rate - All φ Values')  # Set plot title
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend
    axes[2].grid(True, alpha=0.3)  # Enable grid
    
    axes[3].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
    axes[3].set_ylabel('<Z>(t)')  # Set y-label
    axes[3].set_title('<Z>(t) vs Shear Rate - All φ Values')  # Set plot title
    axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend
    axes[3].grid(True, alpha=0.3)  # Enable grid
    
    axes[4].set_xlabel(r'$\dot{\gamma}$')  # Set x-label
    axes[4].set_ylabel('<n>(t) (Fraction of Particles in Frictional Contact)')  # Set y-label
    axes[4].set_title('<n>(t) vs Shear Rate - All φ Values')  # Set plot title
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Add legend
    axes[4].grid(True, alpha=0.3)  # Enable grid
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(output_directory / 'coordination_analysis_combined.png', 
                dpi=600, bbox_inches='tight')  # Save plot with high DPI
    plt.close()  # Close plot
    
    print(f"\nPlots saved to: {output_directory}")

def create_coordination_distribution_plots(results, output_directory):
    """
    Create plots of P(n) vs n for all shear rates and phi values.
    P(n) is the probability density of the fraction of particles in frictional contact.
    
    Parameters:
    -----------
    results : dict
        Results from process_all_phi_values()
    output_directory : str or Path
        Directory to save plots
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    
    # Colors for different phi values
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (phi, phi_results) in enumerate(results.items()):
        color = colors[i]
        
        # Create a figure for this phi value
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f'Probability Distribution of n for φ = {phi}', fontsize=16)
        
        for j, result in enumerate(phi_results):
            # Get n(t) values for all timesteps
            n_values = result['n_t']
            
            # Calculate histogram (P(n) as probability density)
            n_bins = np.linspace(0, 1, 50)  # Bin edges from 0 to 1 for fraction
            hist, bins = np.histogram(n_values, bins=n_bins, density=True)
            
            # Plot P(n) vs n
            ax.plot(bins[:-1] + np.diff(bins)[0]/2, hist, 
                    alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color, marker='o', linestyle='-', markersize=4)
        
        ax.set_xlabel(r'$n(t)$ (Fraction of Particles in Frictional Contact)')  # X-axis: n(t)
        ax.set_ylabel(r'$P(n)$ (Probability Density)')  # Y-axis: P(n)
        ax.set_title('Probability Distribution of n')  # Set plot title
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_directory / f'probability_distribution_n_phi_{phi:.2f}.png', 
                    dpi=600, bbox_inches='tight')  # Highest resolution
        plt.close()
    
    print(f"\nProbability distribution plots saved to: {output_directory}")

def create_summary_table(results, output_directory):
    """
    Create a summary table of final time-averaged values <Z> and <n>.
    """
    output_directory = Path(output_directory)
    
    summary_data = []
    
    for phi, phi_results in results.items():
        for i, result in enumerate(phi_results):
            summary_data.append({
                'phi': phi,
                'run': i + 1,
                'filename': result['filename'],
                '<Z>_final': result['Z_final_avg'],
                '<n>_final': result['n_final_avg'],
                'final_shear_rate': result['shear_rate'][-1],
                'final_viscosity': result['viscosity'][-1]
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_directory / 'coordination_summary.csv', index=False)
    
    # Print summary statistics
    print("\nSummary Statistics (Final Moving Averages):")
    print("=" * 60)
    
    phi_summary = df.groupby('phi').agg({
        '<Z>_final': ['mean', 'std'],
        '<n>_final': ['mean', 'std']
    }).round(4)
    
    print(phi_summary)
    
    return df

# Main execution
if __name__ == "__main__":
    # Set paths - UPDATE THESE PATHS TO MATCH YOUR SYSTEM
    data_directory = Path("/Volumes/T7 Shield/3D Analysis")  # Where coordination analysis files are saved
    output_directory = Path("/Volumes/T7 Shield/3D Analysis/plot 2")  # Changed to "plot 2"
    
    print("Starting Coordination Number Analysis...")
    print("=" * 50)
    print(f"Looking for coordination analysis files in: {data_directory}")
    
    # List all .txt files to help debug file locations
    all_txt_files = list(data_directory.glob("*.txt"))
    recursive_txt_files = list(data_directory.rglob("*.txt"))
    
    print(f"Found {len(all_txt_files)} .txt files in main directory")
    print(f"Found {len(recursive_txt_files)} .txt files total (including subdirectories)")
    
    if recursive_txt_files:
        print("\nSample files found:")
        for file in recursive_txt_files[:5]:  # Show first 5 files
            print(f"  {file}")
        if len(recursive_txt_files) > 5:
            print(f"  ... and {len(recursive_txt_files) - 5} more files")
    
    # Process all data
    results = process_all_phi_values(data_directory)
    
    if not results:
        print("No data files found. Please check the data directory path.")
        exit(1)
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results, output_directory)
    
    # Create coordination distribution plots
    print("\nCreating probability distribution plots...")
    create_coordination_distribution_plots(results, output_directory)
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(results, output_directory)
    
    print("\nAnalysis completed!")
    print(f"Results saved to: {output_directory}")
    
    # Display final summary
    print("\nFinal Summary by φ (Moving Averages at Final Timestep):")
    print("=" * 60)
    for phi, phi_results in results.items():
        Z_final_avg_values = [result['Z_final_avg'] for result in phi_results]
        n_final_avg_values = [result['n_final_avg'] for result in phi_results]
        
        print(f"φ = {phi}:")
        print(f"  <Z> (final) = {np.mean(Z_final_avg_values):.4f} ± {np.std(Z_final_avg_values):.4f}")
        print(f"  <n> (final) = {np.mean(n_final_avg_values):.4f} ± {np.std(n_final_avg_values):.4f}")
        print(f"  Files processed: {len(phi_results)}")