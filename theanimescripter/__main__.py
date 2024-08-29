from pathlib import Path
import os
import sys

os.add_dll_directory(
    str((Path(sys.exec_prefix) / "app_packages/Library/bin").resolve())
)
from theanimescripter.app import main

if __name__ == "__main__":
    main()
