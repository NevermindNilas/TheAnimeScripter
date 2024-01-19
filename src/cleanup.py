import os
import shutil

def delete_pycache(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_folder = os.path.join(root, dir)
                print(f'Deleting {pycache_folder}')
                shutil.rmtree(pycache_folder)

def delete_weights(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth') or file.endswith('.pkl') or file.endswith('.cpkl'):
                weight_file = os.path.join(root, file)
                print(f'Deleting {weight_file}')
                os.remove(weight_file)

def delete_ffmpeg(directory):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(os.path.join(dir_path, 'ffmpeg')):
        print(f'Deleting {os.path.join(dir_path, "ffmpeg")}')
        shutil.rmtree(os.path.join(dir_path, 'ffmpeg'))    

if __name__ == "__main__":
    delete_pycache('./')
    delete_weights('./')
    delete_ffmpeg('./')