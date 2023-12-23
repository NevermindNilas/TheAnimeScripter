from argparse import Namespace
import os
from PyShiftCore import *
from main import *
import traceback

app = App()

def request_from_JS(args_list):
    try:
        # Only arg we need to check is output folder, all others were already verified by JS.
        print("request_from_JS")
        upscale_model, denoise_option, dedup_opt, interp_opt, upscale_opt, upscale_value, interp_value, output_folder = args_list
            
        path = ""   
        
        if dedup_opt == False and interp_opt == False and upscale_opt == False:
            return "You must select at least one option."
        
        layer = app.project.activeLayer     # get the activeLayer
        print(layer.name)
        if layer is None:
            return "No active layer selected."
        
        if isinstance(layer, Layer):
            source = layer.source
            if isinstance(source, FootageItem):
                path = source.path
        print(path)     
        if path == "":
            return "No active footage selected."
        if os.path.exists(output_folder) == False:
            return "Output folder does not exist."
        else:
            #remove ".mp4" from the end of the path variable
            newpath = path[:-4]
            output_folder = output_folder + "/" + os.path.basename(newpath) + "_upscaled.mp4"
       # Define the keys corresponding to the values in args_list
        arg_keys = ["upscale_model", "denoise_option", "dedup_opt", "interp_opt", 
                    "upscale_opt", "upscale_value", "interp_value", "output_folder"]

        # Create a dictionary by zipping arg_keys with args_list
        args_dict = dict(zip(arg_keys, args_list))

        # Add additional attributes to args_dict
        args_dict.update({
            "input": path,  # Presumably set earlier in your code
            "output": output_folder,  # Presumably set earlier in your code
            "interpolate": interp_opt,
            "interpolate_factor": interp_value,
            "upscale": upscale_opt,
            "upscale_factor": upscale_value,
            "upscale_method": upscale_model,
            "cugan_kind": denoise_option,
            "dedup": dedup_opt,
            "dedup_sens": 5,
            "dedup_method": "ffmpeg",
            "nt": 1,
            "half": 1,
            "inpoint": 0,
            "outpoint": 0
        })

        # Unpack the dictionary into an argparse.Namespace
        args = Namespace(**args_dict)
        
        
        process_request(args)
        # to get out path, take the folder, add the name of the file
        out_path = output_folder
        source.replace("new_name", out_path) # Replace the source with the new one
        return "Success!"
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return "Error occurred while processing request."

def process_request(data):
    print("process_request")
    Main(data)
    
    
            
    
    
        
