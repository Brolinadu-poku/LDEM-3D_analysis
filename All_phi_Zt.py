import os  # Import os module for interacting with the operating system (e.g., file paths)
import glob  # Import glob module for finding files matching a pattern (e.g., *.dat)
import platform  # Import platform module to detect the operating system (e.g., Darwin, Linux)
from pathlib import Path  # Import Path class for cross-platform file path handling
import numpy as np  # Import numpy for numerical operations (e.g., arrays, math)
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting data
import matplotlib.colors as mcolors  # Import matplotlib.colors for color handling (not used directly but kept for compatibility)

# Enable LaTeX rendering for plot text (titles, labels) to use mathematical formatting
plt.rcParams['text.usetex'] = True
# Set font family to serif for better compatibility with LaTeX rendering
plt.rcParams['font.family'] = 'serif'
# Set default font size to 12 for consistent text appearance in plots
plt.rcParams['font.size'] = 12

# Get the current operating system (e.g., 'Darwin' for macOS, 'Linux' for Linux)
system_platform = platform.system()
# Define paths based on platform for macOS ('Darwin')
if system_platform == 'Darwin':
    # Set topDir to the directory containing simulation data on an external drive
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 2r")
    # Set output_path to the directory where plots will be saved
    output_path = Path("/Volumes/T7 Shield/3D Analysis")
# Define paths for Linux
elif system_platform == 'Linux':
    # Same topDir as macOS, assuming the external drive is mounted similarly
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 2r")
    # Set output_path to a user-specific Dropbox directory for saving plots
    output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis")
# Raise an error if the OS is neither Darwin nor Linux
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Create the output directory (and any parent directories) if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# Define simulation parameters: list of volume fractions (phi) to analyze
phi = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
# Define aspect ratios (ar), currently a single value
ar = [1.4]
# Define velocity ratios (vr), currently a single value as a string
vr = ['0.5']
# Define run numbers (numRun), currently a single run
numRun = [1]
# Define number of particles in the simulation, from metadata
n_particles = 1000

# Define file patterns for data files using wildcards
particleFile = 'par_*.dat'  # Pattern for particle position files
dataFile = 'data_*.dat'  # Pattern for data files (e.g., shear rate, viscosity)
interactionFile = 'int_*.dat'  # Pattern for interaction files (e.g., contact states)

def read_data_file(file_path):
    """Read data_*.dat file and extract specified columns."""
    # Load the file into a numpy array using loadtxt
    data = np.loadtxt(file_path)
    # Return a dictionary with shear rate (col 3), viscosity (col 4), and time (col 1)
    return {
        'shear_rate': data[:, 2],  # Column 3: shear rate values
        'viscosity': data[:, 3],   # Column 4: viscosity values
        'time': data[:, 0]         # Column 1: time values
    }

def read_particles_file(file_path):
    """Read par_*.dat file and extract particle positions."""
    # Open the file in read mode
    with open(file_path, 'r') as f:
        # Read all lines, skipping the first 22 comment lines
        lines = f.readlines()[22:]
    # Initialize list to store particle data frames
    parList = []
    # Initialize temporary list for current frame
    frame = []
    # Counter for hash (#) lines to detect frame boundaries
    hashCounter = 0
    # Iterate through each line in the file
    for line in lines:
        # Check if line starts with '#', indicating a frame separator
        if line.startswith('#'):
            hashCounter += 1
            # When 7 hash lines are encountered and frame has data, save the frame
            if hashCounter == 7 and frame:
                parList.append(np.array(frame))  # Convert frame to numpy array and store
                frame = []  # Reset frame for next set of data
                hashCounter = 0  # Reset hash counter
        else:
            # Split non-comment line into floats and append to current frame
            frame.append([float(x) for x in line.split()])
    # Append the last frame if it exists
    if frame:
        parList.append(np.array(frame))
    # Return list of frames, each containing x, y, z positions (columns 2, 3, 4)
    return [frame[:, [2, 3, 4]] for frame in parList]

def read_interaction_file(file_path):
    """Read int_*.dat file and extract contact states and particle labels."""
    # Check if file is empty; if so, set number of lines to skip to 0
    nl = 27 if os.path.getsize(file_path) > 0 else 0
    # Open the file in read mode
    with open(file_path, 'r') as f:
        # Read all lines, skipping the first 27 (or 0) comment lines
        lines = f.readlines()[nl:]
    # Initialize list to store interaction data frames
    interactions = []
    # Initialize temporary list for current frame
    temp = []
    # Counter for hash (#) lines to detect frame boundaries
    hashCounter = 0
    # Iterate through each line in the file
    for line in lines:
        # Check if line starts with '#', indicating a frame separator
        if line.startswith('#'):
            hashCounter += 1
            # When 7 hash lines are encountered and temp has data, save the frame
            if hashCounter == 7 and temp:
                interactions.append(np.array(temp))  # Convert temp to numpy array and store
                temp = []  # Reset temp for next set of data
                hashCounter = 0  # Reset hash counter
        else:
            # Split non-comment line into floats and append to current frame
            temp.append([float(x) for x in line.split()])
    # Append the last frame if it exists
    if temp:
        interactions.append(np.array(temp))
    # Return list of interaction frames
    return interactions

# Initialize dictionary to store data for all phi values for combined plotting
all_phi_data = {}

# Define distinct colors for each phi value, matching the number of phi values (8)
colors = ['blue', 'green', 'yellow', 'red', 'brown', 'grey', 'pink', 'violet']

# Iterate over each phi value with its index
for i, phii in enumerate(phi):
    # Format phi value as a string (3 decimals if >2 decimal places, else 2)
    phii_str = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)
    # Initialize dictionary entry for this phi with lists for time, contacts, and assigned color
    all_phi_data[phii] = {'time': [], 'contacts': [], 'color': colors[i]}
    
    # Iterate over each aspect ratio
    for j, arj in enumerate(ar):
        # Iterate over each velocity ratio
        for k, vrk in enumerate(vr):
            # Iterate over each run number
            for m, run in enumerate(numRun):
                # Construct directory path for the simulation data
                dataname = f"{topDir}/phi_{phii_str}/ar_{arj}/Vr_{vrk}/run_{run}"
                # Check if the directory exists; skip if not
                if not os.path.exists(dataname):
                    print(f"Directory {dataname} not found. Skipping...")
                    continue

                # Find all particle files matching the pattern
                par_files = glob.glob(f'{dataname}/{particleFile}')
                # Find all data files matching the pattern
                data_files = glob.glob(f'{dataname}/{dataFile}')
                # Find all interaction files matching the pattern
                interaction_files = glob.glob(f'{dataname}/{interactionFile}')

                # Skip if any file type is missing
                if not par_files or not data_files or not interaction_files:
                    print(f"Missing files in {dataname}. Skipping...")
                    continue

                # Initialize lists to store time and contact data for this phi
                phi_time_data = []
                phi_contact_data = []
                
                # Iterate over each particle file
                for par_file in par_files:
                    # Extract base filename (without 'par_' and '.dat') for matching
                    base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')
                    # Find matching data file by base name
                    data_file = next((f for f in data_files if os.path.basename(f).replace('data_', '').replace('.dat', '') == base_name), None)
                    # Find matching interaction file by base name
                    interaction_file = next((f for f in interaction_files if os.path.basename(f).replace('int_', '').replace('.dat', '') == base_name), None)

                    # Skip if matching data or interaction file is missing
                    if not data_file or not interaction_file:
                        print(f"Missing matching data or interaction file for {par_file}. Skipping...")
                        continue

                    # Read data from the files
                    data_dict = read_data_file(data_file)  # Get shear rate, viscosity, time
                    positions = read_particles_file(par_file)  # Get particle positions
                    interactions = read_interaction_file(interaction_file)  # Get interaction data

                    # Initialize list to store total frictional contacts per timestep
                    total_frictional_contacts = []
                    # Get time values from data file
                    timesteps = data_dict['time']
                    # Get shear rate values from data file
                    shear_rates = data_dict['shear_rate']

                    # Calculate strain as time * shear_rate
                    strains = timesteps * shear_rates

                    # Find minimum length of data arrays to avoid index errors
                    min_length = min(len(positions), len(interactions), len(timesteps))
                    # Process each timestep
                    for t in range(min_length):
                        # Only process timesteps where strain > 1
                        if strains[t] > 1:
                            # Initialize counter for frictional contacts
                            frictional_count = 0
                            # Iterate over interaction rows in the current timestep
                            for row in interactions[t]:
                                # Extract particle IDs and contact state
                                p1, p2, contact_state = int(row[0]), int(row[1]), int(row[10])
                                # Count non-sliding (2) or sliding (3) frictional contacts
                                if contact_state in [2, 3]:
                                    frictional_count += 1

                            # Append the count of frictional contacts
                            total_frictional_contacts.append(frictional_count)
                            # Append the corresponding time
                            phi_time_data.append(timesteps[t])
                            # Append the frictional count to contact data
                            phi_contact_data.append(frictional_count)

                # Store time and contact data for this phi value
                all_phi_data[phii]['time'] = phi_time_data
                all_phi_data[phii]['contacts'] = phi_contact_data

                # Generate individual plot for this phi if data exists
                if phi_time_data and phi_contact_data:
                    # Create a new figure with size 10x6 inches
                    plt.figure(figsize=(10, 6))
                    # Plot time vs frictional contacts with markers and lines, using assigned color
                    plt.plot(phi_time_data, phi_contact_data, 'o-', 
                            color=all_phi_data[phii]['color'], alpha=0.7, markersize=3)
                    # Set x-axis label with LaTeX
                    plt.xlabel(r'Time', fontsize=12)
                    # Set y-axis label with LaTeX
                    plt.ylabel(r'Total Number of Frictional Contacts', fontsize=12)
                    # Set plot title with LaTeX, including phi value and strain condition
                    plt.title(r'Total Frictional Contacts vs Time for $\phi = {}$ (Strain $> 1$)'.format(phii), fontsize=14)
                    # Add a light grid to the plot
                    plt.grid(True, alpha=0.3)
                    # Adjust layout to prevent label cutoff
                    plt.tight_layout()
                    
                    # Save the plot as a PNG file with 300 DPI
                    plt.savefig(output_path / f'frictional_contacts_phi_{phii_str}_strain.png', 
                               dpi=300, bbox_inches='tight')
                    # Close the figure to free memory
                    plt.close()

                # Print confirmation that this phi value was processed
                print(f"Processed phi={phii} in {dataname}. Individual plot generated.")

# Generate combined plot for all phi values
plt.figure(figsize=(12, 8))  # Create a new figure with size 12x8 inches
# Iterate over each phi value
for phii in phi:
    # Check if data exists for this phi
    if all_phi_data[phii]['time'] and all_phi_data[phii]['contacts']:
        # Plot time vs frictional contacts with markers, lines, and LaTeX label, using assigned color
        plt.plot(all_phi_data[phii]['time'], all_phi_data[phii]['contacts'], 
                'o-', color=all_phi_data[phii]['color'], alpha=0.7, 
                label=r'$\phi = {}$'.format(phii), markersize=2)

# Set x-axis label with LaTeX
plt.xlabel(r'Time', fontsize=12)
# Set y-axis label with LaTeX
plt.ylabel(r'Total Number of Frictional Contacts', fontsize=12)
# Set plot title with LaTeX, indicating all phi values and strain condition
plt.title(r'Total Frictional Contacts vs Time for All $\phi$ Values (Strain $> 1$)', fontsize=14)
# Add legend outside the plot area
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# Add a light grid to the plot
plt.grid(True, alpha=0.3)
# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the combined plot as a PNG file with 300 DPI
plt.savefig(output_path / 'frictional_contacts_all_phi_strain.png', 
           dpi=300, bbox_inches='tight')
# Close the figure to free memory
plt.close()

# Print summary statistics for frictional contacts
print("\nSummary Statistics (Strain > 1):")
print("="*50)  # Print a separator line
# Iterate over each phi value
for phii in phi:
    # Check if contact data exists for this phi
    if all_phi_data[phii]['contacts']:
        # Convert contacts to numpy array for calculations
        contacts = np.array(all_phi_data[phii]['contacts'])
        # Print mean, standard deviation, max, and min of contacts
        print(f"φ = {phii:5.2f}: Mean = {np.mean(contacts):6.1f}, "
              f"Std = {np.std(contacts):6.1f}, "
              f"Max = {np.max(contacts):6.0f}, "
              f"Min = {np.min(contacts):6.0f}")

# Print the location where plots are saved
print(f"\nPlots saved to: {output_path}")
# Indicate the naming convention for individual plots
print("Individual plots: frictional_contacts_phi_X.XX_strain.png")
# Indicate the name of the combined plot
print("Combined plot: frictional_contacts_all_phi_strain.png")