# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""

import os
from pathlib import Path
from typing import Union  # from Python 3.10: TypeAlias

# PathLike: TypeAlias = Union[str, Path, os.PathLike]  # from Python 3.10
PathLike = Union[str, Path, os.PathLike]
