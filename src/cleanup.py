import os
import shutil

def delete_files(directory, file_endings=None, dir_names=None):
    """
    A simple script to help me clean up some residue files and directories after testing
    """
    for root, dirs, files in os.walk(directory):
        if file_endings:
            for file in files:
                if any(file.endswith(ending) for ending in file_endings):
                    file_path = os.path.join(root, file)
                    print(f'Deleting {file_path}')
                    os.remove(file_path)
        if dir_names:
            for dir in dirs.copy():
                if dir in dir_names:
                    dir_path = os.path.join(root, dir)
                    print(f'Deleting {dir_path}')
                    shutil.rmtree(dir_path)

    # Use absolute paths for the output directory and log file
    dir_path = os.path.dirname(os.path.abspath(__file__))
    tas_path = os.path.dirname(dir_path)
    
    output_dir = os.path.join(tas_path, 'output')
    log_file = os.path.join(tas_path, 'log.txt')
    if os.path.exists(output_dir):
        print(f'Deleting {output_dir}')
        shutil.rmtree(output_dir)
    if os.path.exists(log_file):
        print(f'Deleting {log_file}')
        os.remove(log_file)
if __name__ == "__main__":
    delete_files('./', file_endings=['.pth', '.pkl', '.cpkl', '.ckpt', ".onnx", ".json"],
                 dir_names=['__pycache__', 'ffmpeg', 'hub', 'weights', 'dist', 'build', 'venv'])