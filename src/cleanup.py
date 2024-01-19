import os
import shutil

def delete_pycache(directory):
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir == '__pycache__':
                pycache_folder = os.path.join(root, dir)
                print(f'Deleting {pycache_folder}')
                shutil.rmtree(pycache_folder)

delete_pycache('./')