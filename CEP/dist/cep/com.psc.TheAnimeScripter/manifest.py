from PyShiftCore import *   
import os

def main():
    manifest = Manifest()
    manifest.name = "TheAnimeScripter"
    manifest.version = "0.0.1"
    manifest.description = "An extension designed for upscaling and denoising footage."
    manifest.author = "TheAnimeScripter"
    #the same didrectory as this specific file
    manifest.entry = os.path.join(os.path.dirname(__file__), "entry.py")

    return manifest

if __name__ == "__main__":
    manifest = main()
