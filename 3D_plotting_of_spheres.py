import os
import glob
import platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Define paths based on platform
system_platform = platform.system()
if system_platform == 'Darwin':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")
    output_path = Path("/Volumes/T7 Shield/3D Analysis 10r")
elif system_platform == 'Linux':
    topDir = Path("/Volumes/T7 Shield/NEW KN VALUES FOR SUSPENSION DYNAMICS/Stress 10r")
    output_path = Path("/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis")
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

def draw_box_wireframe(ax, Lx, Ly, Lz):
    """Draw wireframe box boundaries centered at (0, 0, 0)."""
    half_x, half_y, half_z = Lx / 2, Ly / 2, Lz / 2
    vertices = [
        [-half_x, -half_y, -half_z], [half_x, -half_y, -half_z], [half_x, half_y, -half_z], [-half_x, half_y, -half_z],  # Bottom face
        [-half_x, -half_y, half_z], [half_x, -half_y, half_z], [half_x, half_y, half_z], [-half_x, half_y, half_z]  # Top face
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ]
    for edge in edges:
        points = np.array([vertices[edge[0]], vertices[edge[1]]])
        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.5, linewidth=1)

def create_3d_particle_plot(particles_frame, coordination_numbers, metadata, timestep, 
                           phi_val, save_path=None, show_plot=True):
    """Create a 3D plot of particles colored by coordination number, fitted to box."""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract particle data
    particle_indices = particles_frame[:, 0].astype(int)
    radii = particles_frame[:, 1]
    positions = particles_frame[:, 2:5]  # x, y, z positions
    
    # Debug: Print position ranges to verify scaling
    print(f"Position ranges (x, y, z): min={np.min(positions, axis=0)}, max={np.max(positions, axis=0)}")
    print(f"Box dimensions (Lx, Ly, Lz): {metadata['Lx']}, {metadata['Ly']}, {metadata['Lz']}")
    
    # Use raw positions; adjust limits to center and extend slightly
    Lx, Ly, Lz = metadata['Lx'], metadata['Ly'], metadata['Lz']
    half_x, half_y, half_z = Lx / 2, Ly / 2, Lz / 2
    ax.set_xlim(-half_x - 0.2, half_x + 0.2)
    ax.set_ylim(-half_y - 0.2, half_y + 0.2)
    ax.set_zlim(-half_z - 0.2, half_z + 0.2)
    
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
    
    # Scale particle sizes
    size_scale = 500
    sizes = (radii ** 2) * size_scale
    
    # Create scatter plot
    scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                        c=coordination_numbers, cmap='viridis', 
                        s=sizes, alpha=0.7, edgecolors='black', linewidth=0.1,
                        vmin=0, vmax=max_coord)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Coordination Number (Zi)', fontsize=12)
    
    # Draw box wireframe
    draw_box_wireframe(ax, Lx, Ly, Lz)
    
    # Set labels and title
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_zlabel('Z Position', fontsize=12)
    ax.set_title(f'3D Particle Visualization\nφ = {phi_val}, Timestep = {timestep}\n'
                f'Box: {Lx:.1f} × {Ly:.1f} × {Lz:.1f}', fontsize=14)
    
    # Set aspect ratio and remove grid
    ax.set_box_aspect([1, 1, 1])  # Equal aspect, scaled by limits
    ax.grid(False)
    ax.axis('off')  # Remove axes for clean look
    
    # Add statistics text
    avg_coord = np.mean(coordination_numbers)
    max_coord_actual = np.max(coordination_numbers)
    stats_text = f'Avg Zi: {avg_coord:.2f}\nMax Zi: {max_coord_actual:.0f}\nParticles: {len(particles_frame)}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_3d_animation(particles_data, interactions_data, metadata, phi_val, 
                       output_file, max_frames=50):
    """Create animated 3D visualization of particles over time."""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Box dimensions
    Lx, Ly, Lz = metadata['Lx'], metadata['Ly'], metadata['Lz']
    half_x, half_y, half_z = Lx / 2, Ly / 2, Lz / 2
    
    # Set up the plot limits and labels
    ax.set_xlim(-half_x - 0.2, half_x + 0.2)
    ax.set_ylim(-half_y - 0.2, half_y + 0.2)
    ax.set_zlim(-half_z - 0.2, half_z + 0.2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # No grid
    ax.grid(False)
    ax.axis('off')
    
    # Limit frames for performance
    n_frames = min(len(particles_data), max_frames)
    frame_indices = np.linspace(0, len(particles_data)-1, n_frames, dtype=int)
    
    # Initialize empty scatter plot
    scat = ax.scatter([], [], [], c=[], cmap='viridis', s=[], alpha=0.7)
    
    # Title text
    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=14)
    
    def animate(frame_idx):
        """Animation function for each frame."""
        actual_frame = frame_indices[frame_idx]
        
        # Get particle data for this frame
        particles_frame = particles_data[actual_frame]
        particle_indices = particles_frame[:, 0]
        
        # Calculate coordination numbers
        coord_numbers = calculate_coordination_numbers_frame(interactions_data[actual_frame], particle_indices)
        
        # Extract positions and sizes
        positions = particles_frame[:, 2:5]  # x, y, z
        radii = particles_frame[:, 1]
        sizes = (radii ** 2) * 500  # Scale for visualization
        
        # Clear and redraw
        ax.clear()
        
        # No grid
        ax.grid(False)
        ax.axis('off')
        
        # Draw box wireframe
        draw_box_wireframe(ax, Lx, Ly, Lz)
        
        # Plot particles
        scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                         c=coord_numbers, cmap='viridis', s=sizes, alpha=0.7,
                         edgecolors='black', linewidth=0.1, vmin=0, vmax=10)
        
        # Reset labels and limits
        ax.set_xlim(-half_x - 0.2, half_x + 0.2)
        ax.set_ylim(-half_y - 0.2, half_y + 0.2)
        ax.set_zlim(-half_z - 0.2, half_z + 0.2)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'3D Particle Animation - φ = {phi_val}\nTimestep = {actual_frame}')
        ax.set_box_aspect([1, 1, 1])
        
        return scat,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                  interval=200, blit=False, repeat=True)
    
    # Save animation
    print(f"Creating animation with {n_frames} frames...")
    anim.save(output_file, writer='pillow', fps=5, dpi=150)
    print(f"Animation saved: {output_file}")
    
    plt.close()

def process_3d_visualization(phi_val, timestep_range=None, create_animation_flag=False):
    """Process 3D visualization for a given phi value."""
    
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
                        
                        save_path = output_path / f"3d_particles_phi_{phi_str}_timestep_{t:04d}_{base_name}.png"
                        create_3d_particle_plot(particles_data[t], coord_numbers, metadata, 
                                              t, phi_val, save_path, show_plot=False)
                    
                    # Create animation if requested
                    if create_animation_flag and min_length > 1:
                        anim_path = output_path / f"3d_particles_animation_phi_{phi_str}_{base_name}.gif"
                        create_3d_animation(particles_data, interactions_data, metadata, 
                                          phi_val, anim_path, max_frames=30)

# Main execution functions
def create_static_plots_all_phi(timestep_range=None):
    """Create static 3D plots for all phi values."""
    print("Creating static 3D particle plots...")
    
    for phi_val in phi:
        print(f"\nProcessing phi = {phi_val}")
        process_3d_visualization(phi_val, timestep_range, create_animation_flag=False)

def create_animations_all_phi():
    """Create animations for all phi values."""
    print("Creating 3D particle animations...")
    
    for phi_val in phi:
        print(f"\nCreating animation for phi = {phi_val}")
        process_3d_visualization(phi_val, timestep_range=None, create_animation_flag=True)

def create_single_phi_visualization(phi_val, timesteps=None, animation=False):
    """Create visualization for a single phi value."""
    print(f"Creating 3D visualization for phi = {phi_val}")
    
    timestep_range = None
    if timesteps is not None:
        if isinstance(timesteps, list):
            timestep_range = (min(timesteps), max(timesteps) + 1)
        elif isinstance(timesteps, tuple):
            timestep_range = timesteps
    
    process_3d_visualization(phi_val, timestep_range, animation)

# Example usage and main execution
if __name__ == "__main__":
    print("="*60)
    print("3D PARTICLE VISUALIZATION")
    print("="*60)
    
    # Choose what to run:
    
    # Option 1: Create static plots for first phi value (quick test)
    print("Creating test visualization for phi = 0.52...")
    create_single_phi_visualization(0.52, timesteps=[0, 10, 20], animation=False)
    
    # Option 2: Create static plots for all phi values (uncomment to run)
    # create_static_plots_all_phi(timestep_range=(0, 5))  # First 5 timesteps
    
    # Option 3: Create animations (uncomment to run - takes longer)
    # create_animations_all_phi()
    
    # Option 4: Single phi with animation (uncomment to run)
    # create_single_phi_visualization(0.52, animation=True)
    
    print(f"\n3D visualizations saved to: {output_path}")
    print("Static plots: 3d_particles_phi_X.XX_timestep_XXXX_filename.png")
    print("Animations: 3d_particles_animation_phi_X.XX_filename.gif")