import os
import sys
import subprocess
import os

# Testing the basic functionality of the script
def rife_test(main_py_path):
    for scale in [2, 3, 4]:
        try:
            command = f"python {main_py_path} --video input/test.mp4 --output test_rife_{scale}x.mp4 --model_type rife --multi {scale} "
            subprocess.run([command], shell=True)
        except Exception as e:
            raise e
        
def cugan_test(main_py_path):
    for model_type in ["cugan", "shufflecugan"]:
        if model_type == "shufflecugan":
            for nt in [1, 2, 3, 4]:
                scale = 2 # shufflecugan only supports 2x
                command = f"python {main_py_path} --video input/test.mp4 --output test_{model_type}_{scale}x.mp4 --model_type {model_type} --multi {scale} -nt {nt} "
                subprocess.run([command], shell=True)
        
        else:
            for scale in [2, 3, 4]:
                for nt in [1, 2, 3, 4]:
                    for kind_model in ["no-denoise", "conservative", "denoise1x", "denoise2x"]:
                        command = f"python {main_py_path} --video input/test.mp4 --output test_{model_type}_{scale}x_{kind_model}.mp4 --model_type {model_type} --multi {scale} -nt {nt} --kind_model {kind_model} "
                        subprocess.run([command], shell=True)

if __name__ == "__main__":
    main_file_path = os.path.abspath(__file__)
    main_directory = os.path.dirname(main_file_path)
    main_py_path = os.path.join(main_directory, 'main.py')
    rife_test(main_py_path)
    cugan_test(main_py_path)