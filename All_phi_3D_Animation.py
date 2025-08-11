import os
import subprocess
import platform

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

# Video parameters
framerate = 8
codec = 'libx264'
pix_fmt = 'yuv420p'

# Specific phi values to process
phi_values = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]

# Timestep range: 0 to 2000 with step of 50
timesteps = list(range(0, 2001, 50))

input_folder = TopDir

for phi in phi_values:
    phi_str = f"{phi:.2f}"  # Format phi to 2 decimal places
    output_file = f"Zi_movie_phi{phi_str}.mp4"

    if os.path.exists(input_folder):
        if os.path.exists(os.path.join(OutDir, output_file)):
            print(f'\nMovie already exists - {output_file}\n')
            continue

        # Check if PNG files exist for this phi value and required timesteps
        missing_files = []
        existing_files = []

        for timestep in timesteps:
            filename_pattern = f'vpython_style_3d_phi_{phi_str}_timestep_{timestep:04d}_random_seed_params_stress2r_shear.png'
            file_path = os.path.join(input_folder, filename_pattern)

            if os.path.exists(file_path):
                existing_files.append(filename_pattern)
            else:
                missing_files.append(filename_pattern)

        if not existing_files:
            print(f'\nNo PNG files found for phi={phi_str} in {input_folder}\n')
            continue

        if missing_files:
            print(f'\nWarning: Missing {len(missing_files)} files for phi={phi_str}')
            print(f'Found {len(existing_files)} files out of {len(timesteps)} expected')
            print('Proceeding with available files...\n')

        # Create a temporary file list for ffmpeg
        file_list_path = os.path.join(input_folder, f'temp_filelist_phi_{phi_str}.txt')

        try:
            # Write the list of existing files to a text file
            with open(file_list_path, 'w') as f:
                for timestep in timesteps:
                    filename_pattern = f'vpython_style_3d_phi_{phi_str}_timestep_{timestep:04d}_random_seed_params_stress2r_shear.png'
                    file_path = os.path.join(input_folder, filename_pattern)
                    if os.path.exists(file_path):
                        # Write in format expected by ffmpeg concat demuxer
                        f.write(f"file '{filename_pattern}'\n")
                        f.write(f"duration {1.0 / framerate}\n")  # Duration per frame

            # Add final frame duration
            with open(file_list_path, 'a') as f:
                # Get the last existing file for final frame
                last_file = None
                for timestep in reversed(timesteps):
                    filename_pattern = f'vpython_style_3d_phi_{phi_str}_timestep_{timestep:04d}_random_seed_params_stress2r_shear.png'
                    if os.path.exists(os.path.join(input_folder, filename_pattern)):
                        last_file = filename_pattern
                        break
                if last_file:
                    f.write(f"file '{last_file}'\n")  # Final frame without duration

            # FFmpeg command using concat demuxer
            command = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-r', str(framerate),
                '-c:v', codec,
                '-b:v', '15M',
                '-preset', 'slow',
                '-pix_fmt', pix_fmt,
                os.path.join(OutDir, output_file)
            ]

            print(f"Creating video for phi={phi_str}...")
            subprocess.run(command, check=True)
            print(f"Video created successfully: {output_file}\n")

        except subprocess.CalledProcessError as e:
            print(f"Error during video creation for phi={phi_str}: {e}\n")
        except Exception as e:
            print(f"Unexpected error for phi={phi_str}: {e}\n")
        finally:
            # Clean up temporary file
            if os.path.exists(file_list_path):
                os.remove(file_list_path)
    else:
        print(f"Input folder not found: {input_folder}\n")