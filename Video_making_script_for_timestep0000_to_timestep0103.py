import os
import subprocess
import platform
import re

'''
Aug 08, 2025, 11:15 PM EDT
RVP (modified)

This script makes movies from 3D spherical ball snapshots using ffmpeg.
Input: PNG files in the 3D Analysis directory from the 3D visualization script.
Pre-requisite: PNG files must exist in TopDir.
'''

# Define paths based on platform (matching 3D script)
system_platform = os.name
if system_platform == 'posix':  # macOS or Linux
    if platform.system() == 'Darwin':
        TopDir = "/Volumes/T7 Shield/3D Analysis"
        OutDir = "/Volumes/T7 Shield/3D Analysis"
    else:  # Linux
        TopDir = "/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis"
        OutDir = "/Users/brolinadu-poku/City College Dropbox/Brolin Adu Poku/3D Analysis"
else:
    raise OSError(f"Unsupported OS: {system_platform}")

# Input folder (no subdirectories needed since files are in TopDir)
filename = ['']  # Single empty string to process all files in TopDir

framerate = 8
codec = 'libx264'
pix_fmt = 'yuv420p'

for i in range(len(filename)):
    input_folder = TopDir
    # Group by phi value for separate videos
    phi_values = sorted(set(f.split('phi_')[1].split('_timestep_')[0] for f in os.listdir(input_folder) if
                            f.startswith('vpython_style_3d_') and f.endswith('.png')))

    for phi in phi_values:
        output_file = f"{phi}_animation.mp4"

        if os.path.exists(input_folder):
            if os.path.exists(os.path.join(OutDir, output_file)):
                print(f'\nMovie already exists - {output_file}\n')
            else:
                # Get all PNG files for this phi value
                png_files = [f for f in os.listdir(input_folder) if
                             f.startswith(f'vpython_style_3d_phi_{phi}_') and f.endswith('.png')]

                # Filter files to only include timesteps that are multiples of 50
                filtered_files = []
                for f in png_files:
                    match = re.search(r'timestep_(\d+)', f)
                    if match:
                        timestep = int(match.group(1))
                        if timestep % 50 == 0:
                            filtered_files.append((timestep, f))

                if not filtered_files:
                    print(f'\nNo PNG files found for phi={phi} with timesteps at 50 increments in {input_folder}\n')
                    continue

                # Sort files by timestep number
                filtered_files.sort()
                frame_numbers = [timestep for timestep, f in filtered_files]
                start_number = frame_numbers[0] if frame_numbers else 0

                # Create a temporary directory for selected frames
                temp_dir = os.path.join(input_folder, f'temp_{phi}')
                os.makedirs(temp_dir, exist_ok=True)

                # Create symlinks for selected frames with sequential numbering
                for idx, (timestep, filename) in enumerate(filtered_files):
                    src = os.path.join(input_folder, filename)
                    dst = os.path.join(temp_dir, f'frame_{idx:04d}.png')
                    if not os.path.exists(dst):
                        os.symlink(src, dst)

                # Use the temporary directory with sequential frames for ffmpeg
                command = [
                    'ffmpeg',
                    '-framerate', str(framerate),
                    '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-r', str(framerate),
                    '-c:v', codec,
                    '-b:v', '15M',
                    '-preset', 'slow',
                    '-pix_fmt', pix_fmt,
                    os.path.join(OutDir, output_file)
                ]

                try:
                    subprocess.run(command, check=True)
                    print(f"\nVideo created successfully: {output_file}\n")
                except subprocess.CalledProcessError as e:
                    print(f"\nError during video creation: {e}\n")
                finally:
                    # Clean up temporary directory
                    for f in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, f))
                    os.rmdir(temp_dir)
        else:
            print(f"\nInput folder not found: {input_folder}\n")