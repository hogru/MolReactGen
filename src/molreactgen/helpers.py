# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""
import argparse
import hashlib
import inspect
import json
import logging
import os
import sys
import warnings
from functools import partialmethod
from os import PathLike
from pathlib import Path, PosixPath
from types import FrameType
from typing import Any, Generator, Iterable, Optional, Sequence, Union, overload

import torch
from loguru import logger

LOG_LEVELS = (
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
)
DEFAULT_LOG_LEVEL = logging.INFO
SIGNS_FOR_ROOT_DIR = (".git", "pyproject.toml", "setup.py", "setup.cfg")


###############################################################################
# Tally, used during generation                                             #
###############################################################################


class Tally:
    def __init__(self, counters: Union[Sequence[str], str]) -> None:
        if not isinstance(counters, Iterable):
            counters = [counters]

        counters = [str(c) for c in counters]

        if len(counters) != len(set(counters)):
            raise ValueError("counters must be unique")

        self._counters: dict[str, int] = {str(k): 0 for k in counters}
        self._key_to_idx: dict[str, int] = {
            str(k): i for i, k in enumerate(counters)
        }
        self._idx_to_key: dict[int, str] = {
            i: str(k) for i, k in enumerate(counters)
        }
        counters_one_up = ["_base"]
        counters_one_up.extend(counters[:-1])
        assert len(counters) == len(counters_one_up)
        self._counter_one_up: dict[str, str] = {
            str(counter): str(one_up)
            for counter, one_up in zip(counters, counters_one_up)
        }

    def _get_value_from_idx(self, idx: int) -> int:
        return self._counters[self._idx_to_key[idx]]

    def increment(self, counter: str, increment: int = 1) -> None:
        if counter in self._counters:
            self._counters[counter] += increment
        else:
            raise AttributeError(f"Tally {counter} not found")

    @overload
    def get_count(self) -> dict[str, int]:
        ...

    @overload
    def get_count(self, counter: str) -> int:
        ...

    def get_count(
        self, counter: Optional[str] = None
    ) -> Union[dict[str, int], int]:
        counts: Union[dict[str, int], int]
        if counter is None:
            counts = self._counters
        elif counter in self._counters:
            counts = self._counters[counter]
        else:
            raise AttributeError(f"Tally {counter} not found")

        return counts

    @overload
    def get_absolute_fraction(self) -> dict[str, float]:
        ...

    @overload
    def get_absolute_fraction(self, counter: str) -> float:
        ...

    def get_absolute_fraction(
        self, counter: Optional[str] = None
    ) -> Union[dict[str, float], float]:
        counts = self.get_count()
        base_value = self._get_value_from_idx(0)
        fractions: Union[dict[str, float], float]
        if counter is None:
            if base_value == 0:
                fractions = {k: float("nan") for k in counts}
            else:
                fractions = {k: v / base_value for k, v in counts.items()}
        elif counter in self._counters:
            if base_value == 0:
                fractions = float("nan")
            else:
                fractions = self._counters[counter] / base_value
        else:
            raise AttributeError(f"Tally {counter} not found")

        return fractions

    @overload
    def get_relative_fraction(self) -> dict[str, float]:
        ...

    @overload
    def get_relative_fraction(self, counter: str) -> float:
        ...

    def get_relative_fraction(
        self, counter: Optional[str] = None
    ) -> Union[dict[str, float], float]:
        counts = self.get_count()
        count_names = list(counts.keys())
        count_values = list(counts.values())
        count_names_one_up = ["_base"]
        count_names_one_up.extend(count_names[-1])
        count_values_one_up = [count_values[0]]
        count_values_one_up.extend(count_values[:-1])
        fractions: Union[dict[str, float], float]
        if counter is None:
            fraction_values = [
                c / c_one_up if c_one_up != 0 else float("nan")
                for c, c_one_up in zip(count_values, count_values_one_up)
            ]
            fractions = {k: v for k, v in zip(count_names, fraction_values)}
        elif counter in self._counters:
            idx = self._key_to_idx[counter]
            if count_values_one_up[idx] == 0:
                fractions = float("nan")
            else:
                fractions = count_values[idx] / count_values_one_up[idx]
        else:
            raise AttributeError(f"Tally {counter} not found")

        return fractions

    # TODO implement this
    def _save_to_csv(self, file_path: PathLike[str]) -> None:
        pass

    # TODO implement this
    def _save_to_json(self, file_path: PathLike[str]) -> None:
        pass

    # TODO implement this
    def save_to_file(
        self, file_path: PathLike[str], file_format: str = "json"
    ) -> None:
        file_path = Path(file_path).resolve()
        if file_format.upper() == "CSV":
            self._save_to_csv(file_path)
        elif file_format.upper() == "JSON":
            self._save_to_json(file_path)
        else:
            raise ValueError(f"Unknown format {file_format}")


###############################################################################
# Loguru Logging                                                              #
###############################################################################


def determine_log_level(
    adjustments: Optional[Iterable[int]] = None,
    default_log_level: int = DEFAULT_LOG_LEVEL,
    log_levels: Iterable[int] = LOG_LEVELS,
) -> int:
    adjustments = () if adjustments is None else adjustments
    log_levels = sorted(list(log_levels))
    log_level_idx: int = log_levels.index(default_log_level)
    # For each "-q" and "-v" flag, adjust the logging verbosity accordingly
    # making sure to clamp off the value from 0 to 4, inclusive of both
    for adj in adjustments:
        log_level_idx = min(len(log_levels) - 1, max(log_level_idx + adj, 0))

    return log_levels[log_level_idx]


# Code taken from https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
# Type annotations are my own
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: Union[int, str]
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame: Optional[FrameType]
        depth: int
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def show_warning(message: str, *_: Any, **__: Any) -> None:
    logger.warning(message)


def configure_logging(
    log_level: int = logging.INFO,
    *,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    file_log_level: Optional[int] = None,
    log_dir: Optional[PathLike[str]] = None,
    log_file: Optional[PathLike[str]] = None,
    rotation: str = "1 day",
    retention: str = "7 days",
) -> None:
    # This is the default format used by loguru
    # default_console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | " \
    #                          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " \
    #                          "<level>{message}</level>"
    default_console_format = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    default_file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | "
        "{name}:{function}:{line} - <level>{message}</level>"
    )

    # Allow for separate log levels for console and file logging
    console_log_level: int = int(log_level)
    file_log_level = (
        console_log_level if file_log_level is None else int(file_log_level)
    )

    console_format = (
        default_console_format
        if console_format is None
        else str(console_format)
    )
    file_format = (
        default_file_format if file_format is None else str(file_format)
    )

    # Determine log file path
    # To be changed with python version ≥ 3.11: A list of FrameInfo objects is returned.
    caller_file_path = Path(inspect.stack()[1].filename).resolve()
    if log_dir is None:
        # log_dir = Path(caller_file_path.parent) / "logs"
        log_dir = (
            guess_project_root_dir(caller_file_path=caller_file_path) / "logs"
        )
    else:
        log_dir = Path(log_dir).resolve()
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = Path(str(caller_file_path.stem) + ".log")

    log_file = log_dir / log_file

    rotation = str(rotation)
    retention = str(retention)

    # Remove all previously added handlers
    logger.remove()

    # Allow for logging of headers, add log level
    try:
        logger.level("HEADING", no=21, color="<red><BLACK><bold>")
        logger.__class__.heading = partialmethod(logger.__class__.log, "HEADING")  # type: ignore
    except TypeError:  # log level already exists
        pass

    # Add console handler
    logger.add(sys.stderr, level=console_log_level, format=console_format)

    # Add file handler
    logger.add(
        sink=log_file,
        rotation=rotation,
        retention=retention,
        level=file_log_level,
        format=file_format,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        colorize=False,
    )

    # Redirect warnings to logger
    warnings.showwarning = show_warning  # type: ignore

    # Intercept logging messages from other libraries
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.debug(
        f"Logging with log level {log_level} to {log_file} with rotation {rotation} and retention {retention}"
    )


###############################################################################
# File related                                                                #
###############################################################################


def guess_project_root_dir(
    caller_file_path: Optional[PathLike[str]] = None,
    signs_for_root_dir: Iterable[str] = SIGNS_FOR_ROOT_DIR,
) -> Path:
    """Guess the root directory of the project."""
    caller_file_path = (
        Path(inspect.stack()[1].filename).resolve()
        if caller_file_path is None
        else Path(caller_file_path).resolve()
    )
    directory = caller_file_path
    while (directory != directory.parent) and (directory := directory.parent):
        if any(
            directory.joinpath(sign).exists() for sign in signs_for_root_dir
        ):
            return directory

    return caller_file_path.parent


def get_hash_code(
    file_path: Union[str, Path], algorithm: str = "sha256"
) -> int:
    file_path = Path(file_path).resolve()
    if algorithm.upper() == "MD5":
        hash_fn = hashlib.md5()
    elif algorithm.upper() == "SHA256":
        hash_fn = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm '{algorithm}'")

    file_list: Union[list[Path], Generator[Path, None, None]]
    if file_path.is_file():
        file_list = [file_path]
    elif file_path.is_dir():
        file_list = file_path.rglob("*")
    else:
        raise ValueError(f"Invalid file_path: {file_path}")

    for file in file_list:
        # print(file)
        if file.is_file():
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(hash_fn.block_size), b""):
                    hash_fn.update(chunk)

    # with open(file_path, "rb") as file:
    #     while True:
    #         chunk = file.read(hash_fn.block_size)
    #         if not chunk:
    #             break
    #         hash_fn.update(chunk)

    return int(hash_fn.hexdigest(), 16)


class ArgsEncoder(json.JSONEncoder):
    def default(self, x: Any) -> Any:
        if isinstance(x, PosixPath):
            return x.as_posix()
        else:
            return super().default(x)


def save_commandline_arguments(
    args: argparse.Namespace,
    file_path: PathLike[str],
    keys_to_remove: Optional[Iterable[str]] = None,
) -> None:
    """Save commandline arguments to a file."""
    dict_to_save = dict(args.__dict__)
    if keys_to_remove is not None:
        for key in keys_to_remove:
            dict_to_save.pop(key, None)
    file_path = Path(file_path).resolve()
    logger.debug(f"Saving command-line arguments to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(dict_to_save, f, cls=ArgsEncoder, indent=4, sort_keys=True)


def create_file_link(
    from_file_path: PathLike[str],
    to_file_path: PathLike[str],
    hard_link: bool = False,
) -> None:
    from_file_path = Path(from_file_path)
    to_file_path = Path(to_file_path).resolve()
    if not to_file_path.exists():
        raise FileNotFoundError(f"File {to_file_path} does not exist")
    from_file_path.unlink(missing_ok=True)

    if hard_link:
        # deprecated in 3.10, use from_file_path.hardlink_to(to_file_path) instead
        to_file_path.link_to(from_file_path)
    else:  # symlink
        from_file_path.symlink_to(to_file_path)


###############################################################################
# Miscellaneous                                                               #
###############################################################################


def get_num_workers(spare_cpus: int = 1) -> int:
    if int(spare_cpus) < 0:
        spare_cpus = 0
    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1

    num_workers = max(num_cpus - spare_cpus, 1)
    return num_workers


def get_device() -> torch.device:
    """Determine available device.

    Returns:
        torch.device: device(type='cuda') if cuda is available, device(type='cpu') otherwise.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device(device)
    return torch.device(device)


def get_device_type() -> str:
    """Provides GPU type/name (for information only)

    Returns:
        str: GPU type/name
    """

    device_type = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    return device_type
