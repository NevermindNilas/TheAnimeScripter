import os
import shutil


def delete_files(directory, file_endings=None, dir_names=None):
    for root, dirs, files in os.walk(directory):
        if file_endings:
            for file in files:
                if any(file.endswith(ending) for ending in file_endings):
                    file_path = os.path.join(root, file)
                    print(f'Deleting {file_path}')
                    os.remove(file_path)
        if dir_names:
            for dir in dirs:
                if dir in dir_names:
                    dir_path = os.path.join(root, dir)
                    print(f'Deleting {dir_path}')
                    shutil.rmtree(dir_path)


if __name__ == "__main__":
    delete_files('./', file_endings=['.pth', '.pkl', '.cpkl', '.ckpt'],
                 dir_names=['__pycache__', 'ffmpeg', 'hub'])
