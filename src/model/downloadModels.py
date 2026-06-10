"""Backward-compatibility shim — import from src.model.registry or src.model.download directly.

  src.model.registry  -- weightsBaseDir, weightsDir, modelsList, modelsMap
  src.model.download  -- downloadAndLog, downloadModels, resolveWeightPath
"""

from src.model.registry import weightsBaseDir, weightsDir, modelsList, modelsMap
from src.model.download import downloadAndLog, downloadModels, resolveWeightPath
