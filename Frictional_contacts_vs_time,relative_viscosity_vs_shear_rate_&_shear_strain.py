'''This script plots 
(i) the total Frictional contacts vs time for all phi_values (individually and altogether), 
(ii) the relative viscosity vs shear rate, and 
(iii) the relative viscosity vs shear strain'''

import os
import glob
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Enable LaTeX rendering for plot text
plt.rcParams['text.usetex'] = True
# Set font family to serif for LaTeX compatibility
plt.rcParams['font.family'] = 'serif'
# Set default font size to 12
plt.rcParams['font.size'] = 12

# Get the current operating system
system_platform = platform.system()
# Define paths for macOS
if system_platform == 'Darwin':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 100r")
    base_output_path = Path("/Volumes/T7 Shield/3D Analysis 100r")
# Define paths for Linux
elif system_platform == 'Linux':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 100r")
    base_output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis")
# Raise error for unsupported OS
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Create base output directory if it doesn't exist
base_output_path.mkdir(parents=True, exist_ok=True)

# Define list of volume fractions (phi)
phi = [0.52, 0.53]
# Define aspect ratios (ar) - can be extended to [1.4, 2.0, 4.0] etc.
ar = [1.4]
# Define velocity ratios (vr)
vr = ['0.5']
# Define run numbers
numRun = [1]
# Define number of particles
n_particles = 1000

# Define file patterns for data files
particleFile = 'par_*.dat'
dataFile = 'data_*.dat'
interactionFile = 'int_*.dat'


def create_ar_output_structure(base_path, aspect_ratios):
    """
    Create output directory structure for each aspect ratio
    """
    ar_paths = {}
    for ar_val in aspect_ratios:
        ar_folder_name = f"ar_{ar_val}"
        ar_path = base_path / ar_folder_name
        ar_path.mkdir(parents=True, exist_ok=True)
        ar_paths[ar_val] = ar_path
        print(f"Created/verified directory: {ar_path}")
    return ar_paths


def read_data_file(file_path):
    # Load data file into numpy array
    data = np.loadtxt(file_path)
    # Return dictionary with shear rate, viscosity, and time
    return {
        'shear_rate': data[:, 2],
        'viscosity': data[:, 3],
        'time': data[:, 0]
    }


def read_particles_file(file_path):
    # Open particle file and skip first 22 comment lines
    with open(file_path, 'r') as f:
        lines = f.readlines()[22:]
    # Initialize list for particle data frames
    parList = []
    # Initialize temporary list for current frame
    frame = []
    # Counter for hash lines to detect frame boundaries
    hashCounter = 0
    # Iterate through lines
    for line in lines:
        # Check for frame separator
        if line.startswith('#'):
            hashCounter += 1
            # Save frame after 7 hash lines
            if hashCounter == 7 and frame:
                parList.append(np.array(frame))
                frame = []
                hashCounter = 0
        else:
            # Append data line as floats
            frame.append([float(x) for x in line.split()])
    # Append last frame if it exists
    if frame:
        parList.append(np.array(frame))
    # Return list of position arrays (x, y, z)
    return [frame[:, [2, 3, 4]] for frame in parList]


def read_interaction_file(file_path):
    # Set number of lines to skip (27 if file not empty, else 0)
    nl = 27 if os.path.getsize(file_path) > 0 else 0
    # Open interaction file and skip comment lines
    with open(file_path, 'r') as f:
        lines = f.readlines()[nl:]
    # Initialize list for interaction data frames
    interactions = []
    # Initialize temporary list for current frame
    temp = []
    # Counter for hash lines to detect frame boundaries
    hashCounter = 0
    # Iterate through lines
    for line in lines:
        # Check for frame separator
        if line.startswith('#'):
            hashCounter += 1
            # Save frame after 7 hash lines
            if hashCounter == 7 and temp:
                interactions.append(np.array(temp))
                temp = []
                hashCounter = 0
        else:
            # Append data line as floats
            temp.append([float(x) for x in line.split()])
    # Append last frame if it exists
    if temp:
        interactions.append(np.array(temp))
    # Return list of interaction frames
    return interactions


def generate_individual_plots(phi_data, phi_val, ar_val, output_path):
    """
    Generate individual plots for a specific phi and aspect ratio
    """
    phii_str = '{:.3f}'.format(phi_val) if len(str(phi_val).split('.')[1]) > 2 else '{:.2f}'.format(phi_val)

    # Generate individual frictional contacts plot if data exists
    if phi_data['time'] and phi_data['contacts']:
        plt.figure(figsize=(10, 6))
        plt.plot(phi_data['time'], phi_data['contacts'], 'o-',
                 color=phi_data['color'], alpha=0.8, markersize=3)
        plt.xlabel(r'Time', fontsize=12)
        plt.ylabel(r'Total Number of Frictional Contacts', fontsize=12)
        title_str = r'Total Frictional Contacts vs Time for $\phi = ' + str(phi_val) + r'$, AR = ' + str(
            ar_val) + r' (Strain $> 1$)'
        plt.title(title_str, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'frictional_contacts_phi_{phii_str}_ar_{ar_val}_strain.png',
                    dpi=600, bbox_inches='tight')
        plt.close()

    # Generate individual viscosity vs shear rate plot if data exists
    if phi_data['shear_rate'] and phi_data['viscosity']:
        plt.figure(figsize=(10, 6))
        plt.plot(phi_data['shear_rate'], phi_data['viscosity'], 'o-',
                 color=phi_data['color'], alpha=0.8, markersize=3)
        plt.xlabel(r'Shear Rate $\dot{\gamma}$', fontsize=12)
        plt.ylabel(r'Relative Viscosity $\eta_r$', fontsize=12)
        title_str = r'Relative Viscosity $\eta_r$ vs Shear Rate $\dot{\gamma}$ for $\phi = ' + str(
            phi_val) + r'$, AR = ' + str(ar_val) + r' (Strain $> 1$)'
        plt.title(title_str, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'viscosity_vs_shear_rate_phi_{phii_str}_ar_{ar_val}.png',
                    dpi=600, bbox_inches='tight')
        plt.close()

    # Generate individual viscosity vs strain plot if data exists
    if phi_data['strain'] and phi_data['viscosity']:
        plt.figure(figsize=(10, 6))
        plt.plot(phi_data['strain'], phi_data['viscosity'], 'o-',
                 color=phi_data['color'], alpha=0.8, markersize=3)
        plt.xlabel(r'Cumulated Shear Strain $\gamma$', fontsize=12)
        plt.ylabel(r'Relative Viscosity $\eta_r$', fontsize=12)
        title_str = r'Relative Viscosity $\eta_r$ vs Cumulated Shear Strain $\gamma$ for $\phi = ' + str(
            phi_val) + r'$, AR = ' + str(ar_val) + r' (Strain $> 1$)'
        plt.title(title_str, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / f'viscosity_vs_strain_phi_{phii_str}_ar_{ar_val}.png',
                    dpi=600, bbox_inches='tight')
        plt.close()


def generate_combined_plots(all_phi_data, ar_val, output_path):
    """
    Generate combined plots for all phi values for a specific aspect ratio
    """
    # Generate combined frictional contacts plot
    plt.figure(figsize=(12, 8))
    for phii in phi:
        if all_phi_data[phii]['time'] and all_phi_data[phii]['contacts']:
            plt.plot(all_phi_data[phii]['time'], all_phi_data[phii]['contacts'],
                     'o-', color=all_phi_data[phii]['color'], alpha=0.8,
                     label=rf'$\phi = {phii}$', markersize=2)

    plt.xlabel(r'Time', fontsize=12)
    plt.ylabel(r'Total Number of Frictional Contacts', fontsize=12)
    title_str = r'Total Frictional Contacts vs Time for All $\phi$ Values, AR = ' + str(ar_val) + r' (Strain $> 1$)'
    plt.title(title_str, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'frictional_contacts_all_phi_ar_{ar_val}_strain.png',
                dpi=600, bbox_inches='tight')
    plt.close()

    # Generate combined viscosity vs shear rate plot
    plt.figure(figsize=(12, 8))
    for phii in phi:
        if all_phi_data[phii]['shear_rate'] and all_phi_data[phii]['viscosity']:
            plt.plot(all_phi_data[phii]['shear_rate'], all_phi_data[phii]['viscosity'],
                     'o-', color=all_phi_data[phii]['color'], alpha=0.8,
                     label=rf'$\phi = {phii}$', markersize=2)

    plt.xlabel(r'Shear Rate $\dot{\gamma}$', fontsize=12)
    plt.ylabel(r'Relative Viscosity $\eta_r$', fontsize=12)
    title_str = r'Relative Viscosity $\eta_r$ vs Shear Rate $\dot{\gamma}$ for All $\phi$ Values, AR = ' + str(
        ar_val) + r' (Strain $> 1$)'
    plt.title(title_str, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'viscosity_vs_shear_rate_all_phi_ar_{ar_val}.png',
                dpi=600, bbox_inches='tight')
    plt.close()

    # Generate combined viscosity vs cumulated shear strain plot
    plt.figure(figsize=(12, 8))
    for phii in phi:
        if all_phi_data[phii]['strain'] and all_phi_data[phii]['viscosity']:
            plt.plot(all_phi_data[phii]['strain'], all_phi_data[phii]['viscosity'],
                     'o-', color=all_phi_data[phii]['color'], alpha=0.8,
                     label=rf'$\phi = {phii}$', markersize=2)

    plt.xlabel(r'Cumulated Shear Strain $\gamma$', fontsize=12)
    plt.ylabel(r'Relative Viscosity $\eta_r$', fontsize=12)
    title_str = r'Relative Viscosity $\eta_r$ vs Cumulated Shear Strain $\gamma$ for All $\phi$ Values, AR = ' + str(
        ar_val) + r' (Strain $> 1$)'
    plt.title(title_str, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / f'viscosity_vs_strain_all_phi_ar_{ar_val}.png',
                dpi=600, bbox_inches='tight')
    plt.close()


# Create output directory structure for each aspect ratio
ar_output_paths = create_ar_output_structure(base_output_path, ar)

# Define distinct colors for each phi
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'lime']

# Main processing loop - iterate over aspect ratios first
for j, arj in enumerate(ar):
    print(f"\nProcessing aspect ratio: {arj}")
    print("=" * 50)

    # Get the output path for this aspect ratio
    current_output_path = ar_output_paths[arj]

    # Initialize dictionary for all phi data for this aspect ratio
    all_phi_data = {}

    # Initialize data dictionary for each phi
    for idx, phii in enumerate(phi):
        all_phi_data[phii] = {
            'time': [], 'contacts': [], 'shear_rate': [],
            'viscosity': [], 'strain': [], 'color': colors[idx]
        }

    # Iterate over phi values
    for i, phii in enumerate(phi):
        phii_str = '{:.3f}'.format(phii) if len(str(phii).split('.')[1]) > 2 else '{:.2f}'.format(phii)

        # Iterate over velocity ratios
        for k, vrk in enumerate(vr):
            # Iterate over run numbers
            for m, run in enumerate(numRun):
                # Construct directory path
                dataname = f"{topDir}/phi_{phii_str}/ar_{arj}/Vr_{vrk}/run_{run}"

                # Skip if directory doesn't exist
                if not os.path.exists(dataname):
                    print(f"Directory {dataname} not found. Skipping...")
                    continue

                # Find particle, data, and interaction files
                par_files = glob.glob(f'{dataname}/{particleFile}')
                data_files = glob.glob(f'{dataname}/{dataFile}')
                interaction_files = glob.glob(f'{dataname}/{interactionFile}')

                # Skip if any file type is missing
                if not par_files or not data_files or not interaction_files:
                    print(f"Missing files in {dataname}. Skipping...")
                    continue

                # Initialize lists for time, contact, shear rate, viscosity, and strain data
                phi_time_data = []
                phi_contact_data = []
                phi_shear_rate_data = []
                phi_viscosity_data = []
                phi_strain_data = []

                # Iterate over particle files
                for par_file in par_files:
                    # Extract base filename for matching
                    base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')
                    # Find matching data file
                    data_file = next((f for f in data_files if
                                      os.path.basename(f).replace('data_', '').replace('.dat', '') == base_name), None)
                    # Find matching interaction file
                    interaction_file = next((f for f in interaction_files if
                                             os.path.basename(f).replace('int_', '').replace('.dat', '') == base_name),
                                            None)

                    # Skip if matching files are missing
                    if not data_file or not interaction_file:
                        print(f"Missing matching data or interaction file for {par_file}. Skipping...")
                        continue

                    # Read data from files
                    data_dict = read_data_file(data_file)
                    positions = read_particles_file(par_file)
                    interactions = read_interaction_file(interaction_file)

                    # Initialize list for frictional contacts
                    total_frictional_contacts = []
                    # Get time values
                    timesteps = data_dict['time']
                    # Get shear rate values
                    shear_rates = data_dict['shear_rate']
                    # Get viscosity values
                    viscosities = data_dict['viscosity']

                    # Calculate strain
                    strains = timesteps * shear_rates

                    # Find minimum data length
                    min_length = min(len(positions), len(interactions), len(timesteps))
                    # Process each timestep
                    for t in range(min_length):
                        # Filter for strain > 1
                        if strains[t] > 1:
                            # Initialize frictional contact counter
                            frictional_count = 0
                            # Iterate over interaction rows
                            for row in interactions[t]:
                                # Extract particle IDs and contact state
                                p1, p2, contact_state = int(row[0]), int(row[1]), int(row[10])
                                # Count frictional contacts (states 2 or 3)
                                if contact_state in [2, 3]:
                                    frictional_count += 1
                            # Append contact count
                            total_frictional_contacts.append(frictional_count)
                            # Append time
                            phi_time_data.append(timesteps[t])
                            # Append contact count to data
                            phi_contact_data.append(frictional_count)
                            # Append shear rate
                            phi_shear_rate_data.append(shear_rates[t])
                            # Append viscosity
                            phi_viscosity_data.append(viscosities[t])
                            # Append strain
                            phi_strain_data.append(strains[t])

                # Store data for this phi
                all_phi_data[phii]['time'] = phi_time_data
                all_phi_data[phii]['contacts'] = phi_contact_data
                all_phi_data[phii]['shear_rate'] = phi_shear_rate_data
                all_phi_data[phii]['viscosity'] = phi_viscosity_data
                all_phi_data[phii]['strain'] = phi_strain_data

                # Generate individual plots for this phi and aspect ratio
                if phi_time_data and phi_contact_data:
                    generate_individual_plots(all_phi_data[phii], phii, arj, current_output_path)
                    print(f"Generated individual plots for phi={phii}, ar={arj}")

    # Generate combined plots for all phi values for this aspect ratio
    generate_combined_plots(all_phi_data, arj, current_output_path)
    print(f"Generated combined plots for ar={arj}")

    # Print summary statistics for frictional contacts for this aspect ratio
    print(f"\nSummary Statistics for Frictional Contacts (AR = {arj}, Strain > 1):")
    print("=" * 60)
    for phii in phi:
        if all_phi_data[phii]['contacts']:
            contacts = np.array(all_phi_data[phii]['contacts'])
            print(f"φ = {phii:5.2f}: Mean = {np.mean(contacts):6.1f}, "
                  f"Std = {np.std(contacts):6.1f}, "
                  f"Max = {np.max(contacts):6.0f}, "
                  f"Min = {np.min(contacts):6.0f}")

    # Print summary statistics for relative viscosity for this aspect ratio
    print(f"\nSummary Statistics for Relative Viscosity (AR = {arj}, Strain > 1):")
    print("=" * 60)
    for phii in phi:
        if all_phi_data[phii]['viscosity']:
            viscosities = np.array(all_phi_data[phii]['viscosity'])
            print(f"φ = {phii:5.2f}: Mean = {np.mean(viscosities):6.1f}, "
                  f"Std = {np.std(viscosities):6.1f}, "
                  f"Max = {np.max(viscosities):6.1f}, "
                  f"Min = {np.min(viscosities):6.1f}")

    print(f"\nPlots for AR = {arj} saved to: {current_output_path}")

# Print final summary
print(f"\n" + "=" * 70)
print("PROCESSING COMPLETE")
print("=" * 70)
print(f"Base output directory: {base_output_path}")
print("Aspect ratio folders created:")
for ar_val, path in ar_output_paths.items():
    print(f"  AR {ar_val}: {path}")
print("\nPlot types generated for each aspect ratio:")
print("  - Individual plots: frictional_contacts_phi_X.XX_ar_X.X_strain.png")
print("  - Individual plots: viscosity_vs_shear_rate_phi_X.XX_ar_X.X.png")
print("  - Individual plots: viscosity_vs_strain_phi_X.XX_ar_X.X.png")
print("  - Combined plots: frictional_contacts_all_phi_ar_X.X_strain.png")
print("  - Combined plots: viscosity_vs_shear_rate_all_phi_ar_X.X.png")
print("  - Combined plots: viscosity_vs_strain_all_phi_ar_X.X.png")