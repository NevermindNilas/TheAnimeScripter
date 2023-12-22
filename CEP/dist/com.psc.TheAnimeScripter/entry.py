import os
from PyShiftCore import *
from main import *

app = App()

def request_from_JS(upscale_model, denoise_option, dedup_opt,
                    interp_opt, upscale_opt, upscale_value, interp_value, output_folder):
    
    # Only arg we need to check is output folder, all others were already verified by JS.
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path = ""   
    
    if dedup_opt == False and interp_opt == False and upscale_opt == False:
        return "You must select at least one option."
    
    layer = app.project.activeLayer     # get the activeLayer
    
    if layer is None:
        return "No active layer selected."
    
    if isinstance(layer, Layer):
        source = layer.source
        if isinstance(source, FootageItem):
            path = source.path
            
    if path == "":
        return "No active footage selected."
    
    args = {
        "input": path,
        "output": output_folder,
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
    }
    
    process_request(args)
    # to get out path, take the folder, add the name of the file
    out_path = os.path.join(output_folder, os.path.basename(path))  # Or however things are named
    source.replace("new_name", out_path) # Replace the source with the new one
    return "Success!"


def process_request(data):
    Main(data)
    
    
            
    
    
        
