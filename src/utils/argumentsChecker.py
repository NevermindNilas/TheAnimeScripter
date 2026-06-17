"""
Backward-compatibility shim. Import from src.cli.* directly.

  src.cli.parser    -- argparse classes, _buildParser, createParser, _add*Options
  src.cli.validator -- argumentsChecker, validation helpers
  src.cli.startup   -- _handleDependencies, _promptDownloadRequirementsSelection
"""

from src.cli.parser import (
    DidYouMeanArgumentParser,
    TASHelpFormatter,
    _addDedupOptions,
    _addDepthOptions,
    _addEncodingOptions,
    _addInterpolationOptions,
    _addMiscOptions,
    _addMotionBlurOptions,
    _addSceneDetectionOptions,
    _addSegmentationOptions,
    _addUpscalingOptions,
    _addVideoProcessingOptions,
    _buildParser,
    _listMethods,
    _supportsColorStdout,
    capabilityMethods,
    createParser,
    logAndPrint,
    str2bool,
)
from src.cli.startup import (
    _handleDependencies,
    _promptDownloadRequirementsSelection,
)
from src.cli.validator import (
    _adjustMethodsBasedOnCuda,
    _autoEnableParentFlags,
    _configureProcessingSettings,
    _downloadOfflineModels,
    _handleDepthSettings,
    _loadJsonConfig,
    _validateCustomUpscaleModel,
    argumentsChecker,
    isAnyOtherProcessingMethodEnabled,
    processURL,
)

__all__ = [
    "logAndPrint",
    "_supportsColorStdout",
    "str2bool",
    "DidYouMeanArgumentParser",
    "TASHelpFormatter",
    "capabilityMethods",
    "_listMethods",
    "_buildParser",
    "createParser",
    "_addInterpolationOptions",
    "_addUpscalingOptions",
    "_addDedupOptions",
    "_addVideoProcessingOptions",
    "_addMotionBlurOptions",
    "_addSegmentationOptions",
    "_addSceneDetectionOptions",
    "_addDepthOptions",
    "_addEncodingOptions",
    "_addMiscOptions",
    "isAnyOtherProcessingMethodEnabled",
    "argumentsChecker",
    "_autoEnableParentFlags",
    "_handleDepthSettings",
    "_configureProcessingSettings",
    "_validateCustomUpscaleModel",
    "_adjustMethodsBasedOnCuda",
    "processURL",
    "_downloadOfflineModels",
    "_loadJsonConfig",
    "_handleDependencies",
    "_promptDownloadRequirementsSelection",
]
