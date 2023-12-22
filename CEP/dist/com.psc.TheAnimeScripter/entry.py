import os
from PyShiftCore import *

app = App()

def request_from_JS(upscale_model, denoise_option, dedup_opt,
                    interp_opt, upscale_opt, upscale_value, interp_value, output_folder):
    
    # all options will be valid, except possibly output_folder, so verify that and create it if necessary
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    path = ""   
    
    # dedupt opt, interp opt, upscale opt, all may be selected ,or only a few, but at least one must be selected
    if dedup_opt == False and interp_opt == False and upscale_opt == False:
        return "You must select at least one option."
    # get the activeLayer
    layer = app.project.activeLayer
    
    if layer is None:
        return "No active layer selected."
    
    if isinstance(layer, Layer):
        source = layer.source
        if isinstance(source, FootageItem):
            path = source.path
            
    if path == "":
        return "No active footage selected."
    
    #formate everything into a dictionary
    request = {
        "path": path,
        "upscale_model": upscale_model,
        "denoise_option": denoise_option,
        "dedup_opt": dedup_opt,
        "interp_opt": interp_opt,
        "upscale_opt": upscale_opt,
        "upscale_value": upscale_value,
        "interp_value": interp_value,
        "output_folder": output_folder
    }
    
    process_request(request)
    app.reportInfo("Refresh the project, and you're set!")
    return "Success!"


def process_request(data):
    case = data["upscale_model"]
    if case == "ShuffleCugan":
        pass
    elif case == "Compact":
        pass
    elif case == "UltraCompact":
        pass
    elif case == "SuperUltraCompact":
        pass
    elif case == "Cugan":
        pass
    elif case == "Cugan-amd":
        pass
    
    
    
    
            
    
    
        
