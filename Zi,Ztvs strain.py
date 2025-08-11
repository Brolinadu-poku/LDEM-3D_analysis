import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Set plotting style (using matplotlib built-in styles)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['text.usetex'] = False  # Use Matplotlib's LaTeX rendering for Greek letters
plt.rcParams['mathtext.fontset'] = 'stix'  # Use STIX fonts for math symbols

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
    filepath = Path(filepath)
    
    # Read metadata from header
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'Volume Fraction' in line:
                metadata['phi'] = float(line.split(':')[1].strip())
            elif 'Aspect Ratio' in line:
                metadata['aspect_ratio'] = float(line.split(':')[1].strip())
            elif 'Velocity Ratio' in line:
                metadata['velocity_ratio'] = line.split(':')[1].strip()
            elif 'Run Number' in line:
                metadata['run_number'] = int(line.split(':')[1].strip())
            elif 'Number of Particles' in line:
                metadata['n_particles'] = int(line.split(':')[1].strip())
            elif 'Box Dimensions' in line:
                dims = line.split('-')[1].strip()
                for dim in dims.split(', '):
                    key, value = dim.split(': ')
                    try:
                        metadata[key] = float(value)
                    except ValueError:
                        metadata[key] = value
    
    # Read numerical data
    data = np.loadtxt(filepath)
    
    # Column mappings
    n_particles = metadata.get('n_particles', 1000)
    columns = {
        'timestep_index': 0,
        'time': 1,
        'shear_rate': 2,
        'viscosity': 3,
        'frictional_contact_number': 4
    }
    
    # Add coordination number columns
    for i in range(n_particles):
        columns[f'coordination_number_particle_{i}'] = 5 + i
    
    return {
        'data': data,
        'metadata': metadata,
        'columns': columns
    }

def calculate_coordination_statistics(analysis_result):
    """
    Calculate Z(t), n(t), <Z>(t), and <n>(t) from coordination analysis data.
    Now using moving time averages and t=1 to T indexing.
    Also calculates cumulative shear strain.
    
    Parameters:
    -----------
    analysis_result : dict
        Result from read_coordination_analysis_file()
        
    Returns:
    --------
    dict : Dictionary containing calculated statistics
    """
    data = analysis_result['data']
    n_particles = analysis_result['metadata']['n_particles']
    
    # Extract coordination numbers for all particles (columns 5 onwards)
    coord_numbers = data[:, 5:5+n_particles]  # Shape: (n_timesteps, n_particles)
    
    # Calculate Z(t) - average coordination number at each timestep (t=1 to T)
    Z_t = np.mean(coord_numbers, axis=1)  # Shape: (n_timesteps,)
    
    # Calculate n(t) - fraction of particles with frictional contacts at each timestep (t=1 to T)
    particles_with_contacts = coord_numbers > 0  # Boolean array
    n_t = np.mean(particles_with_contacts, axis=1)  # Shape: (n_timesteps,)
    
    # Calculate moving time averages <Z>(t) and <n>(t)
    # For timestep t, average from timestep 1 to timestep t
    Z_moving_avg = np.zeros_like(Z_t)  # Moving average of Z
    n_moving_avg = np.zeros_like(n_t)  # Moving average of n
    
    for t in range(len(Z_t)):  # t goes from 0 to T-1 in array indexing (corresponds to timesteps 1 to T)
        # Moving average from timestep 1 to current timestep (t+1)
        Z_moving_avg[t] = np.mean(Z_t[:t+1])  # Average from index 0 to t (timesteps 1 to t+1)
        n_moving_avg[t] = np.mean(n_t[:t+1])  # Average from index 0 to t (timesteps 1 to t+1)
    
    # Final time averages (values at the last timestep)
    Z_final_avg = Z_moving_avg[-1]  # <Z> at final timestep T
    n_final_avg = n_moving_avg[-1]  # <n> at final timestep T
    
    # Extract other relevant data
    time = data[:, 1]
    shear_rate = data[:, 2]
    viscosity = data[:, 3]
    
    # Calculate cumulative shear strain
    # Cumulative strain = integral of shear rate over time
    # Using trapezoidal integration
    dt = np.diff(time)
    cumulative_strain = np.zeros_like(time)
    
    for i in range(1, len(time)):
        # Trapezoidal rule: area = (dt) * (rate[i-1] + rate[i]) / 2
        cumulative_strain[i] = cumulative_strain[i-1] + dt[i-1] * (shear_rate[i-1] + shear_rate[i]) / 2
    
    return {
        'Z_t': Z_t,                        # Average coordination number at each timestep
        'n_t': n_t,                        # Fraction of particles with contacts at each timestep
        'Z_moving_avg': Z_moving_avg,      # Moving time-averaged coordination number <Z>(t)
        'n_moving_avg': n_moving_avg,      # Moving time-averaged fraction <n>(t)
        'Z_final_avg': Z_final_avg,        # Final time-averaged coordination number <Z>
        'n_final_avg': n_final_avg,        # Final time-averaged fraction <n>
        'time': time,                      # Time array (t=1 to T)
        'shear_rate': shear_rate,          # Shear rate array (t=1 to T)
        'cumulative_strain': cumulative_strain,  # Cumulative shear strain array
        'viscosity': viscosity,            # Viscosity array (t=1 to T)
        'coord_numbers': coord_numbers     # Full coordination number matrix
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
    data_directory = Path(data_directory)
    phi_values = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
    
    results = {}
    
    for phi in phi_values:
        phi_str = f"{phi:.2f}"
        print(f"Processing phi = {phi} ({phi_str})...")
        
        # Find coordination analysis files for this phi value
        # Search recursively in case files are in subdirectories
        pattern = f"coordination_analysis_phi_{phi_str}_*.txt"
        files = list(data_directory.glob(pattern))  # Direct search
        
        # If no files found in main directory, search recursively
        if not files:
            files = list(data_directory.rglob(pattern))  # Recursive search
        
        # Also try searching with different phi formatting (e.g., 0.520 vs 0.52)
        if not files:
            phi_str_alt = f"{phi:.3f}"
            pattern_alt = f"coordination_analysis_phi_{phi_str_alt}_*.txt"
            files = list(data_directory.glob(pattern_alt))
            if not files:
                files = list(data_directory.rglob(pattern_alt))
        
        if not files:
            print(f"  No files found for phi = {phi} (searched patterns: {pattern} and coordination_analysis_phi_{phi:.3f}_*.txt)")
            print(f"  Searched in: {data_directory}")
            continue
            
        phi_results = []
        
        for file_path in files:
            try:
                # Read and analyze file
                analysis_result = read_coordination_analysis_file(file_path)
                stats = calculate_coordination_statistics(analysis_result)
                
                # Add metadata
                stats['metadata'] = analysis_result['metadata']
                stats['filename'] = file_path.name
                
                phi_results.append(stats)
                print(f"  Processed: {file_path.name}")
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        if phi_results:
            results[phi] = phi_results
            
            # Print summary statistics for this phi
            all_Z_final_avg = [result['Z_final_avg'] for result in phi_results]
            all_n_final_avg = [result['n_final_avg'] for result in phi_results]
            
            print(f"  Summary for phi = {phi}:")
            print(f"    <Z> (final): {np.min(all_Z_final_avg):.3f} - {np.max(all_Z_final_avg):.3f}")
            print(f"    <n> (final): {np.min(all_n_final_avg):.3f} - {np.max(all_n_final_avg):.3f}")
            print(f"    Number of files: {len(phi_results)}")
    
    return results

def create_plots(results, output_directory):
    """
    Create plots for viscosity, Z(t), n(t), and moving averages vs cumulative shear strain.
    
    Parameters:
    -----------
    results : dict
        Results from process_all_phi_values()
    output_directory : str or Path
        Directory to save plots
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)
    
    # Colors for different phi values (using matplotlib colormap)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Create individual plots for each phi value
    for i, (phi, phi_results) in enumerate(results.items()):
        color = colors[i]
        
        # Create subplots for this phi value (5 plots in a single row)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Coordination Analysis for φ = {phi}', fontsize=16)
        
        for j, result in enumerate(phi_results):
            # Plot 1: Viscosity vs Cumulative Shear Strain
            axes[0].plot(result['cumulative_strain'], result['viscosity'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)
            axes[0].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
            axes[0].set_ylabel('Viscosity')
            axes[0].set_title('Viscosity vs Cumulative Shear Strain')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Z(t) vs Cumulative Shear Strain
            axes[1].plot(result['cumulative_strain'], result['Z_t'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)
            axes[1].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
            axes[1].set_ylabel('Z(t)')
            axes[1].set_title('Instantaneous Coordination Number vs Cumulative Shear Strain')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: n(t) vs Cumulative Shear Strain
            axes[2].plot(result['cumulative_strain'], result['n_t'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)
            axes[2].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
            axes[2].set_ylabel('n(t)')
            axes[2].set_title('Fraction with Contacts vs Cumulative Shear Strain')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Plot 4: <Z>(t) vs Cumulative Shear Strain
            axes[3].plot(result['cumulative_strain'], result['Z_moving_avg'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)
            axes[3].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
            axes[3].set_ylabel('<Z>(t)')
            axes[3].set_title('Moving Averaged Coordination Number vs Cumulative Shear Strain')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            
            # Plot 5: <n>(t) vs Cumulative Shear Strain
            axes[4].plot(result['cumulative_strain'], result['n_moving_avg'], 
                         alpha=0.7, linewidth=1, label=f"Run {j+1}", color=color)
            axes[4].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
            axes[4].set_ylabel('<n>(t)')
            axes[4].set_title('Moving Averaged Fraction vs Cumulative Shear Strain')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_directory / f'coordination_analysis_phi_{phi:.2f}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create combined plots showing all phi values
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('Coordination Analysis - All φ Values', fontsize=16)
    
    for i, (phi, phi_results) in enumerate(results.items()):
        color = colors[i % len(colors)]  # Cycle through colors if needed
        
        # Average across all runs for this phi (if multiple runs exist)
        all_cumulative_strain = []
        all_viscosities = []
        all_Z_t = []
        all_n_t = []
        all_Z_moving_avg = []
        all_n_moving_avg = []
        
        for result in phi_results:
            all_cumulative_strain.append(result['cumulative_strain'])
            all_viscosities.append(result['viscosity'])
            all_Z_t.append(result['Z_t'])
            all_n_t.append(result['n_t'])
            all_Z_moving_avg.append(result['Z_moving_avg'])
            all_n_moving_avg.append(result['n_moving_avg'])
        
        # If multiple runs, average them (assuming same timesteps)
        if len(phi_results) > 1:
            cumulative_strain = np.mean(all_cumulative_strain, axis=0)
            viscosity = np.mean(all_viscosities, axis=0)
            Z_t = np.mean(all_Z_t, axis=0)
            n_t = np.mean(all_n_t, axis=0)
            Z_moving_avg = np.mean(all_Z_moving_avg, axis=0)
            n_moving_avg = np.mean(all_n_moving_avg, axis=0)
        else:
            cumulative_strain = all_cumulative_strain[0]
            viscosity = all_viscosities[0]
            Z_t = all_Z_t[0]
            n_t = all_n_t[0]
            Z_moving_avg = all_Z_moving_avg[0]
            n_moving_avg = all_n_moving_avg[0]
        
        # Plot combined results
        axes[0].plot(cumulative_strain, viscosity, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='o', markersize=2)
        axes[1].plot(cumulative_strain, Z_t, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='s', markersize=2)
        axes[2].plot(cumulative_strain, n_t, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='^', markersize=2)
        axes[3].plot(cumulative_strain, Z_moving_avg, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='d', markersize=2)
        axes[4].plot(cumulative_strain, n_moving_avg, color=color, 
                     linewidth=2, label=f'φ = {phi}', marker='v', markersize=2)
    
    # Set labels and formatting for combined plots
    axes[0].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
    axes[0].set_ylabel('Viscosity')
    axes[0].set_title('Viscosity vs Cumulative Shear Strain - All φ')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
    axes[1].set_ylabel('Z(t)')
    axes[1].set_title('Instantaneous Coordination Number vs Cumulative Shear Strain - All φ')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
    axes[2].set_ylabel('n(t)')
    axes[2].set_title('Fraction with Contacts vs Cumulative Shear Strain - All φ')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
    axes[3].set_ylabel('<Z>(t)')
    axes[3].set_title('Moving Averaged Coordination Number vs Cumulative Shear Strain - All φ')
    axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[3].grid(True, alpha=0.3)
    
    axes[4].set_xlabel(r'$\gamma$ (Cumulative Shear Strain)')
    axes[4].set_ylabel('<n>(t)')
    axes[4].set_title('Moving Averaged Fraction vs Cumulative Shear Strain - All φ')
    axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_directory / 'coordination_analysis_combined.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_directory}")

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
                'final_cumulative_strain': result['cumulative_strain'][-1],
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
        '<n>_final': ['mean', 'std'],
        'final_cumulative_strain': ['mean', 'std']
    }).round(4)
    
    print(phi_summary)
    
    return df

# Main execution
if __name__ == "__main__":
    # Set paths - UPDATE THESE PATHS TO MATCH YOUR SYSTEM
    data_directory = Path("/Volumes/T7 Shield/3D Analysis 10r")  # Where coordination analysis files are saved
    output_directory = Path("/Volumes/T7 Shield/3D Analysis 10r/plot 2")  # Changed to "plot 2"
    
    print("Starting Coordination Number Analysis with Cumulative Shear Strain...")
    print("=" * 70)
    print(f"Looking for coordination analysis files in: {data_directory}")
    print(f"Output will be saved to: {output_directory}")
    
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
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(results, output_directory)
    
    print("\nAnalysis completed!")
    print(f"Results saved to: {output_directory}")
    
    # Display final summary
    print("\nFinal Summary by φ (Moving Averages at Final Timestep):")
    print("=" * 70)
    for phi, phi_results in results.items():
        Z_final_avg_values = [result['Z_final_avg'] for result in phi_results]
        n_final_avg_values = [result['n_final_avg'] for result in phi_results]
        final_strain_values = [result['cumulative_strain'][-1] for result in phi_results]
        
        print(f"φ = {phi}:")
        print(f"  <Z> (final) = {np.mean(Z_final_avg_values):.4f} ± {np.std(Z_final_avg_values):.4f}")
        print(f"  <n> (final) = {np.mean(n_final_avg_values):.4f} ± {np.std(n_final_avg_values):.4f}")
        print(f"  Final strain = {np.mean(final_strain_values):.4f} ± {np.std(final_strain_values):.4f}")
        print(f"  Files processed: {len(phi_results)}")