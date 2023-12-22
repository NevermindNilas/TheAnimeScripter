#PyShiftCore

from enum import Enum
from typing import *

app = App()

class Quality(Enum):
    BEST = int 
    DRAFT = int
    WIREFRAME = int
    NONE = int
    
class App:
    """
    The After Effects application object.  Accessible via the global variable `app`.
    
    --------------------
    Read-Only Properties:
    --------------------
    project: Project
        The currently open project.  Returns null if no project is open.
        
    Methods:
    --------------------
    beginUndoGroup(undo_name: str = "Default Undo Group Name") -> None
        Begins a new undo group.  Undo groups are used to combine several actions into a single undoable operation.
    endUndoGroup() -> None
        Ends the current undo group.  See `beginUndoGroup`.
    reportInfo(info: Any) -> None
        Displays a modal alert box with the given message.
        
    --------------------
    Read-only Properties:
    --------------------
    
    --------------------
    Read-only Methods:
    --------------------
    """
    def __init__(self): ...
    project: Project
    def beginUndoGroup(self, undo_name: str = "Default Undo Group Name") -> None: ...
    def endUndoGroup(self) -> None: ...
    def reportInfo(self, info: Any) -> None: ...

class Project:
    """
    The After Effects project object.  Accessible via the global variable `app.project`.
    
    --------------------
    Read-Only Properties:
    --------------------
    activeItem: Item
        The currently active item. Returns null if no item is active.
    activeLayer: Layer
        The currently active layer. Returns null if no layer is active.
    name: str
        The name of the project.
    path: str
        The full path to the project file.
    items: ItemCollection
        A collection of all items in the project.
    selectedItems: ItemCollection
        A collection of all selected items in the project.
    Methods:
    --------------------
    saveAs(path: str) -> None
        Saves the project to the given path.
        
    --------------------
    Read-only Properties:
    --------------------

    --------------------
    Read-only Methods:
    --------------------
    
    
    """
    def __init__(self): ...
    activeItem: Item
    activeLayer: Layer
    name: str
    path: str
    items: ItemCollection
    selectedItems: ItemCollection
    def saveAs(self, path: str) -> None: ...
    
class Item:
    """
    The base class for all items in After Effects.  Do not instantiate this class directly.
    
    --------------------
    Properties:
    --------------------
    name: str
        The name of the item.
    width: int
        The width of the item.
    height: int
        The height of the item.
    duration: int
        The duration of the item in seconds.
    time: int
        The current time of the item in seconds.
        
    """
    def __init__(self): ...
    name: str
    width: int
    height: int
    duration: int
    time: int
    selected: bool
    
class FolderItem(Item):
    """
    The base class for all folder items in After Effects.  Do not instantiate this class directly.
    
    --------------------
    Read-Only Properties:
    --------------------
    children: ItemCollection
        A collection of all items in the folder.
        
        
    """
    def __init__(self, name: str): ...
    children: ItemCollection
        
class CompItem(Item):
    """
    The After Effects composition object.  Accessible via the global variable `app.project.activeItem`.
    
    --------------------
    Properties:
    --------------------
    selectedLayers: LayerCollection
        A collection of all selected layers in the composition.
    selectedLayer: LayerCollection
        A collection of all selected layers in the composition.
    frameRate: float
        The frame rate of the composition.
    time: float
        The current time of the composition in # of frames.
    duration: float
        The duration of the composition in # of frames.
        
    --------------------
    Read-only Properties:
    --------------------
    layer: LayerCollection
        A collection of all layers in the composition.
    layers: LayerCollection
        A collection of all layers in the composition.
    numLayers: int
        The number of layers in the composition.
        
    Methods:
    TODO: ADD METHODS
    --------------------
    duplicate() -> CompItem
        Duplicates the composition.
    delete() -> None
        Deletes the composition.
        
    """
    def __init__(self, name: str = "Comp", width: int = 1920,
                 height: int = 1080, frameRate: float = 24.0,
                 duration: float = 10.0, aspectRation: float = 1.0): ...
    layer: LayerCollection
    layers: LayerCollection
    selectedLayers: LayerCollection
    selectedLayer: LayerCollection
    numLayers: int
    frameRate: float
    
class FootageItem(Item):
    """
    The base class for all footage items in After Effects.  If Instantiating this class directly, provide ALL arguments, aside from index.

    --------------------
    Read-only Properties:
    --------------------
    path: str
        The full path to the footage file.
        
    
    """
    def __init__(self, name: str = "New Footage", path: str = "", index: int = 1): ...
    path = str
    
class SolidItem(FootageItem):
    """
    The After Effects solid object.  Instantiated via 'SolidItem(name = "New Solid", width = 1920,
                height = 1080, red = 0, green = 0, blue = 0, alpha = 0, duration = 10.0, index = 1)'.
        RGBA values are all floating point values from [0.00, 1.00].
    
    """
    def __init__(self, name: str = "New Solid", width: int = 1920,
                    height: int = 1080, red: int = 0, green: int = 0,
                    blue: int = 0, alpha: int = 0, duration: float = 10.0, index: int = 1): ...
    
class ItemCollection:
    """
    The base class for all item collections in After Effects.  Do not instantiate this class directly.
    
    --------------------
    Read-only Properties:
    --------------------
    length: int
        The number of items in the collection.
        
    Methods:
    --------------------
    __getitem__(key: int) -> Item
        Returns the item at the given index.
    __setitem__(key: int, value: Item) -> None
        Sets the item at the given index.
    __len__() -> int
        Returns the number of items in the collection.
    __iter__() -> Iterator[Item]
        Returns an iterator over the collection.
    """
    def __init__(self): ...
    def __getitem__(self, key: int) -> Item: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Item]: ...
    def append(self, item: Item) -> None: ...
    def remove(self, item: Item) -> None: ...

class Layer:
    """
    The base class for all layers in After Effects.  Do not instantiate this class directly.
    This is one of the largest classes. Only the main properties and methods listed. 
    See API docs or full list of properties and flags.
    
    --------------------
    Properties:
    --------------------
    name: str
        The name of the layer.
    sourceName: str
        The name of the source item.
    time: float
        The current time of the layer in seconds.
    compTime: float
        The current time of the layer in # of frames.
    inPoint: float
        The in point of the layer in seconds.
    compInPoint: float
        The in point of the layer in # of frames.
    duration: float
        The duration of the layer in seconds.
    compDuration: float
        The duration of the layer in # of frames.
    source: Item
        The source item of the layer. Typically a FootageItem.
    """
    def __init__(self): ...
    name: str
    quality: Quality
    startTime: float
    index: int
    video_active: bool
    audio_active: bool
    efects_active: bool
    motion_blur: bool
    frame_blending: bool
    locked: bool
    shy: bool
    collapse: bool
    auto_orient_rotation: bool
    adjustment_layer: bool
    time_remapping: bool
    layer_is_3d: bool
    look_at_camera: bool
    look_at_point: bool
    solo: bool
    markers_locked: bool
    null_layer: bool
    hide_locked_masks: bool
    guide_layer: bool
    advanced_frame_blending: bool
    sublayers_render_separately: bool
    environment_layer: bool
    
    #read only
    sourceName: str
    time: float
    compTime: float
    inPoint: float
    compInPoint: float
    duration: float
    compDuration: float
    source: Item
    
    def duplicate(self) -> Layer: ...
    def delete(self) -> None: ...

class AdjustmentLayer(Layer):
    """
    Extends Layer. Provides Simple interface to create an adjustment layer.
    """
    def __init__(self, comp: CompItem, name: str = "Adjustment Layer"): ...
        
class LayerCollection:
    """
    
    The base class for all layer collections in After Effects.  Do not instantiate this class directly.
    
    --------------------
    Read-only Properties:
    --------------------
    length: int
        The number of layers in the collection.
        
    Methods:
    --------------------
    __getitem__(key: int) -> Layer
        Returns the layer at the given index.
    __setitem__(key: int, value: Layer) -> None
        Sets the layer at the given index.
    __len__() -> int
        Returns the number of layers in the collection.
    __iter__() -> Iterator[Layer]
        Returns an iterator over the collection.
    append(layer: Layer) -> None
        Adds a layer to the end of the collection.
    remove(layer: Layer) -> None
        Removes a layer from the collection.
    pop(index: int) -> Layer
        Removes a layer from the collection and returns it.
    insert(index: int, layer: Layer) -> None
        Inserts a layer into the collection at the given index.
      
    """
    def __init__(self, comp: CompItem, layers: List[Layer]): ...
    def __getitem__(self, key: int) -> Layer: ...
    def __setitem__(self, key: int, value: Layer) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Layer]: ...
    def append(self, layer: Layer) -> None: ...
    def remove(self, layer: Layer) -> None: ... 
    def pop(self, index: int) -> Layer: ...
    def insert(self, index: int, layer: Layer) -> None: ...
    
class Manifest:
    """
    The Manifest class.  Used to define the properties of custom functions written for CEP Extensions.
    
    --------------------
    Properties:
    --------------------
    name: str
        The name of the extension. You will use this same name to call the extension from the CEP side.
    version: str
        The version of the extension.
    author: str
        The author of the extension.
    description: str
        The description of the extension.
    entry: str
        The path to the entry point of the extension. Entry should be name "entry.py". Do not change this line.

    """
    def __init__(self): ...
    name: str
    version: str
    author: str
    description: str
    entry: str
