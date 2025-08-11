import os
import glob
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LightSource
import matplotlib.cm as cm

# Define paths based on platform
system_platform = platform.system()
if system_platform == 'Darwin':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")
    output_path = Path("/Volumes/T7 Shield/3D Analysis 10r")
elif system_platform == 'Linux':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")
    output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis 10r")
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Create output directory if it doesn't exist
output_path.mkdir(parents=True, exist_ok=True)

# Simulation parameters
phi = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
ar = [1.4]
vr = ['0.5']
numRun = [1]
n_particles = 1000

# File patterns
particleFile = 'par_*.dat'
interactionFile = 'int_*.dat'

def create_sphere(center, radius, resolution=15):
    """Create high-quality sphere surface coordinates."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def create_box_faces(Lx, Ly, Lz, alpha=0.1):
    """Create transparent box faces."""
    half_x, half_y, half_z = Lx / 2, Ly / 2, Lz / 2
    
    faces = []
    
    # Bottom face (z = -half_z)
    xx, yy = np.meshgrid([-half_x, half_x], [-half_y, half_y])
    zz = np.full_like(xx, -half_z)
    faces.append((xx, yy, zz))
    
    # Top face (z = half_z)
    xx, yy = np.meshgrid([-half_x, half_x], [-half_y, half_y])
    zz = np.full_like(xx, half_z)
    faces.append((xx, yy, zz))
    
    # Front face (y = -half_y)
    xx, zz = np.meshgrid([-half_x, half_x], [-half_z, half_z])
    yy = np.full_like(xx, -half_y)
    faces.append((xx, yy, zz))
    
    # Back face (y = half_y)
    xx, zz = np.meshgrid([-half_x, half_x], [-half_z, half_z])
    yy = np.full_like(xx, half_y)
    faces.append((xx, yy, zz))
    
    # Left face (x = -half_x)
    yy, zz = np.meshgrid([-half_y, half_y], [-half_z, half_z])
    xx = np.full_like(yy, -half_x)
    faces.append((xx, yy, zz))
    
    # Right face (x = half_x)
    yy, zz = np.meshgrid([-half_y, half_y], [-half_z, half_z])
    xx = np.full_like(yy, half_x)
    faces.append((xx, yy, zz))
    
    return faces

def read_particles_file_3d(file_path):
    """Read par_*.dat file and extract particle data for 3D visualization."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata from header
    metadata = {}
    for line in lines[:22]:
        if line.startswith('# np '):
            metadata['np'] = int(line.split()[-1])
        elif line.startswith('# VF '):
            metadata['VF'] = float(line.split()[-1])
        elif line.startswith('# Lx '):
            metadata['Lx'] = float(line.split()[-1])
        elif line.startswith('# Ly '):
            metadata['Ly'] = float(line.split()[-1])
        elif line.startswith('# Lz '):
            metadata['Lz'] = float(line.split()[-1])
    
    # Process particle data
    lines = lines[22:]  # Skip header
    parList = []
    frame = []
    hashCounter = 0
    
    for line in lines:
        if line.startswith('#'):
            hashCounter += 1
            if hashCounter == 7 and frame:
                parList.append(np.array(frame))
                frame = []
                hashCounter = 0
        else:
            frame.append([float(x) for x in line.split()])
    
    if frame:
        parList.append(np.array(frame))
    
    return parList, metadata

def read_interaction_file_3d(file_path):
    """Read int_*.dat file for coordination numbers."""
    with open(file_path, 'r') as f:
        lines = f.readlines()[27:]  # Skip header
    
    interactions = []
    temp = []
    hashCounter = 0
    
    for line in lines:
        if line.startswith('#'):
            hashCounter += 1
            if hashCounter == 7 and temp:
                interactions.append(np.array(temp))
                temp = []
                hashCounter = 0
        else:
            temp.append([float(x) for x in line.split()])
    
    if temp:
        interactions.append(np.array(temp))
    
    return interactions

def calculate_coordination_numbers_frame(interaction_frame, particle_indices):
    """Calculate coordination numbers for a single frame."""
    max_particle_id = int(np.max(particle_indices)) if len(particle_indices) > 0 else 0
    coordination_numbers = np.zeros(max_particle_id + 1)
    
    if len(interaction_frame) > 0:
        for row in interaction_frame:
            p1, p2, contact_state = int(row[0]), int(row[1]), int(row[10])
            if contact_state in [2, 3]:  # Frictional contacts
                if p1 <= max_particle_id and p2 <= max_particle_id:
                    coordination_numbers[p1] += 1
                    coordination_numbers[p2] += 1
    
    return coordination_numbers[particle_indices.astype(int)]

def create_photorealistic_particle_plot(particles_frame, coordination_numbers, metadata, timestep, 
                                       phi_val, save_path=None, show_plot=True, sphere_resolution=20, 
                                       max_particles_for_spheres=2000, force_spheres=True):
    """Create a photorealistic 3D plot of particles with proper lighting and materials."""
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract particle data
    particle_indices = particles_frame[:, 0].astype(int)
    radii = particles_frame[:, 1]
    positions = particles_frame[:, 2:5]  # x, y, z positions
    
    # Box dimensions
    Lx, Ly, Lz = metadata['Lx'], metadata['Ly'], metadata['Lz']
    half_x, half_y, half_z = Lx / 2, Ly / 2, Lz / 2
    
    # Set plot limits with some padding
    padding = 0.3
    ax.set_xlim(-half_x - padding, half_x + padding)
    ax.set_ylim(-half_y - padding, half_y + padding)
    ax.set_zlim(-half_z - padding, half_z + padding)
    
    # Ensure coordination_numbers matches the number of particles
    if len(coordination_numbers) != len(particles_frame):
        print(f"Warning: Coordination numbers ({len(coordination_numbers)}) don't match particles ({len(particles_frame)})")
        if len(coordination_numbers) > len(particles_frame):
            coordination_numbers = coordination_numbers[:len(particles_frame)]
        else:
            padded_coords = np.zeros(len(particles_frame))
            padded_coords[:len(coordination_numbers)] = coordination_numbers
            coordination_numbers = padded_coords
    
    # Color map for coordination numbers
    coordination_numbers = np.array(coordination_numbers)
    coordination_numbers = np.nan_to_num(coordination_numbers, nan=0.0, posinf=0.0, neginf=0.0)
    max_coord = np.max(coordination_numbers) if np.max(coordination_numbers) > 0 else 1
    
    # Create colormap normalization
    norm = Normalize(vmin=0, vmax=max_coord)
    cmap = cm.viridis
    
    # Set up lighting
    light = LightSource(azdeg=315, altdeg=45)
    
    # Decide rendering method based on particle count and force_spheres flag
    use_3d_spheres = force_spheres or len(particles_frame) <= max_particles_for_spheres
    
    # Optimize sphere resolution based on particle count for performance
    if len(particles_frame) > 500:
        sphere_resolution = max(8, sphere_resolution // 2)  # Reduce resolution for many particles
    elif len(particles_frame) > 1000:
        sphere_resolution = max(6, sphere_resolution // 3)  # Further reduce for very many particles
    
    print(f"Rendering {len(particles_frame)} particles using {'optimized 3D spheres (res={sphere_resolution})' if use_3d_spheres else '2D scatter'}...")
    
    if use_3d_spheres:
        # Render particles as optimized solid 3D spheres
        print(f"Creating {len(particles_frame)} solid 3D spheres with resolution {sphere_resolution}...")
        
        # Batch process spheres for better performance
        for i, (pos, radius, coord_num) in enumerate(zip(positions, radii, coordination_numbers)):
            # Show progress for large numbers of particles
            if len(particles_frame) > 500 and i % 100 == 0:
                print(f"  Processed {i}/{len(particles_frame)} spheres...")
            
            # Create sphere surface
            x_sphere, y_sphere, z_sphere = create_sphere(pos, radius, resolution=sphere_resolution)
            
            # Get base color for this coordination number
            base_color = cmap(norm(coord_num))
            
            # Create solid sphere with optimized settings
            surface = ax.plot_surface(x_sphere, y_sphere, z_sphere, 
                                    color=base_color, 
                                    alpha=0.95,  # Slightly more opaque for solid look
                                    shade=True, 
                                    lightsource=light,
                                    antialiased=False if len(particles_frame) > 500 else True,  # Disable AA for performance
                                    edgecolor='none',
                                    rcount=sphere_resolution, 
                                    ccount=sphere_resolution,
                                    linewidth=0)  # No edge lines for solid appearance
        
        # Create transparent box
        box_faces = create_box_faces(Lx, Ly, Lz)
        for i, (xx, yy, zz) in enumerate(box_faces):
            # Alternate face colors for better visibility
            face_color = 'lightblue' if i % 2 == 0 else 'lightgray'
            ax.plot_surface(xx, yy, zz, alpha=0.05, color=face_color, shade=False)
            
        # Add box edges for definition
        # Draw wireframe edges more prominently
        vertices = [
            [-half_x, -half_y, -half_z], [half_x, -half_y, -half_z], 
            [half_x, half_y, -half_z], [-half_x, half_y, -half_z],  # Bottom face
            [-half_x, -half_y, half_z], [half_x, -half_y, half_z], 
            [half_x, half_y, half_z], [-half_x, half_y, half_z]  # Top face
        ]
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.3, linewidth=1.5)
        
        # Create a dummy scatter plot for colorbar (invisible)
        dummy_scatter = ax.scatter([], [], [], c=[], cmap='viridis', 
                                 vmin=0, vmax=max_coord, alpha=0)
        
    else:
        print(f"Too many particles ({len(particles_frame)}). Using enhanced 2D scatter...")
        
        # Enhanced scatter plot with better visual appearance
        size_scale = 800
        sizes = (radii ** 2) * size_scale
        
        # Create scatter plot with enhanced appearance
        dummy_scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                            c=coordination_numbers, cmap='viridis', 
                            s=sizes, alpha=0.8, edgecolors='black', linewidth=0.5,
                            vmin=0, vmax=max_coord, depthshade=True)
        
        # Draw box wireframe
        vertices = [
            [-half_x, -half_y, -half_z], [half_x, -half_y, -half_z], 
            [half_x, half_y, -half_z], [-half_x, half_y, -half_z],  # Bottom face
            [-half_x, -half_y, half_z], [half_x, -half_y, half_z], 
            [half_x, half_y, half_z], [-half_x, half_y, half_z]  # Top face
        ]
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.4, linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(dummy_scatter, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label('Coordination Number (Zi)', fontsize=14, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    # Set labels and title with better styling
    ax.set_xlabel('X Position', fontsize=14, labelpad=10)
    ax.set_ylabel('Y Position', fontsize=14, labelpad=10)
    ax.set_zlabel('Z Position', fontsize=14, labelpad=10)
    
    rendering_type = f"Solid 3D Spheres (res={sphere_resolution})" if use_3d_spheres else "Enhanced 2D Scatter"
    ax.set_title(f'3D Particle Visualization ({rendering_type})\n'
                f'φ = {phi_val}, Timestep = {timestep}\n'
                f'Box: {Lx:.1f} × {Ly:.1f} × {Lz:.1f}', 
                fontsize=16, pad=20)
    
    # Enhanced visual settings
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.grid(False)
    
    # Set better background and lighting
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make pane edges more subtle
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Better tick styling
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    # Add statistics text with better styling
    avg_coord = np.mean(coordination_numbers)
    max_coord_actual = np.max(coordination_numbers)
    stats_text = f'Average Zi: {avg_coord:.2f}\nMax Zi: {max_coord_actual:.0f}\nParticles: {len(particles_frame)}\nVolume Fraction: {metadata.get("VF", "N/A"):.3f}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved photorealistic 3D plot: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def process_3d_visualization(phi_val, timestep_range=None, create_animation_flag=False,
                            sphere_resolution=15, max_particles_for_spheres=2000, force_spheres=True):
    """Process photorealistic 3D visualization for a given phi value."""
    
    phi_str = '{:.3f}'.format(phi_val) if len(str(phi_val).split('.')[1]) > 2 else '{:.2f}'.format(phi_val)
    
    for j, arj in enumerate(ar):
        for k, vrk in enumerate(vr):
            for m, run in enumerate(numRun):
                dataname = f"{topDir}/phi_{phi_str}/ar_{arj}/Vr_{vrk}/run_{run}"
                
                if not os.path.exists(dataname):
                    print(f"Directory {dataname} not found. Skipping...")
                    continue

                par_files = glob.glob(f'{dataname}/{particleFile}')
                interaction_files = glob.glob(f'{dataname}/{interactionFile}')

                if not par_files or not interaction_files:
                    print(f"Missing files in {dataname}. Skipping...")
                    continue

                for par_file in par_files:
                    base_name = os.path.basename(par_file).replace('par_', '').replace('.dat', '')
                    interaction_file = next((f for f in interaction_files if 
                                           os.path.basename(f).replace('int_', '').replace('.dat', '') == base_name), None)

                    if not interaction_file:
                        print(f"Missing interaction file for {par_file}. Skipping...")
                        continue

                    print(f"Processing: {base_name} for phi = {phi_val}")
                    
                    # Read data
                    particles_data, metadata = read_particles_file_3d(par_file)
                    interactions_data = read_interaction_file_3d(interaction_file)
                    
                    min_length = min(len(particles_data), len(interactions_data))
                    
                    if timestep_range is None:
                        # Create plots for first, middle, and last timesteps
                        timesteps_to_plot = [0, min_length//2, min_length-1]
                    else:
                        start, end = timestep_range
                        timesteps_to_plot = list(range(max(0, start), min(min_length, end)))
                    
                    # Create individual timestep plots
                    for t in timesteps_to_plot:
                        if t >= min_length:
                            continue
                            
                        # Get particle indices for this timestep
                        particle_indices = particles_data[t][:, 0]
                        coord_numbers = calculate_coordination_numbers_frame(interactions_data[t], particle_indices)
                        
                        save_path = output_path / f"solid_3d_spheres_phi_{phi_str}_timestep_{t:04d}_{base_name}.png"
                        create_photorealistic_particle_plot(particles_data[t], coord_numbers, metadata, 
                                                          t, phi_val, save_path, show_plot=False,
                                                          sphere_resolution=sphere_resolution,
                                                          max_particles_for_spheres=max_particles_for_spheres,
                                                          force_spheres=force_spheres)

# Main execution functions
def create_photorealistic_plots_all_phi(timestep_range=None, sphere_resolution=15, max_particles_for_spheres=2000, force_spheres=True):
    """Create solid 3D sphere plots for all phi values."""
    print("Creating solid 3D particle sphere plots...")
    
    for phi_val in phi:
        print(f"\nProcessing phi = {phi_val}")
        process_3d_visualization(phi_val, timestep_range, create_animation_flag=False,
                                sphere_resolution=sphere_resolution,
                                max_particles_for_spheres=max_particles_for_spheres,
                                force_spheres=force_spheres)

def create_single_phi_photorealistic(phi_val, timesteps=None, sphere_resolution=15, max_particles_for_spheres=2000, force_spheres=True):
    """Create solid 3D sphere visualization for a single phi value."""
    print(f"Creating solid 3D sphere visualization for phi = {phi_val}")
    
    timestep_range = None
    if timesteps is not None:
        if isinstance(timesteps, list):
            timestep_range = (min(timesteps), max(timesteps) + 1)
        elif isinstance(timesteps, tuple):
            timestep_range = timesteps
    
    process_3d_visualization(phi_val, timestep_range, create_animation_flag=False,
                            sphere_resolution=sphere_resolution,
                            max_particles_for_spheres=max_particles_for_spheres,
                            force_spheres=force_spheres)

# Example usage and main execution
if __name__ == "__main__":
    print("="*70)
    print("SOLID 3D PARTICLE SPHERES - HIGH VOLUME RENDERING")
    print("="*70)
    
    # Create solid 3D spheres for 1000+ particles
    print("Creating solid 3D spheres for phi = 0.52 (handles 1000+ particles)...")
    create_single_phi_photorealistic(0.52, timesteps=[0, 10, 20], 
                                    sphere_resolution=12,  # Balanced quality/performance
                                    max_particles_for_spheres=5000,  # Allow up to 5000 particles
                                    force_spheres=True)  # Always use spheres
    
    # Option 2: Create solid sphere plots for all phi values (uncomment to run)
    # create_photorealistic_plots_all_phi(timestep_range=(0, 5), 
    #                                    sphere_resolution=10, 
    #                                    max_particles_for_spheres=5000, 
    #                                    force_spheres=True)
    
    print(f"\nSolid 3D sphere visualizations saved to: {output_path}")
    print("Files: solid_3d_spheres_phi_X.XX_timestep_XXXX_filename.png")
    print("\nOptimizations for large particle counts:")
    print("- Adaptive sphere resolution (lower res for >500 particles)")
    print("- Disabled anti-aliasing for >500 particles (performance)")
    print("- Progress tracking for large batches")
    print("- Solid appearance with optimized lighting")
    print("- Handles 1000+ particles as true 3D spheres!")