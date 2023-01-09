# coding=utf-8
"""
Auto-Regressive Molecule and Reaction Template Generator
Causal language modeling (CLM) with a transformer decoder model
Author: Stephan Holzgruber
Student ID: K08608294
"""


import hashlib
import inspect
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Generator, Iterable, Optional, Sequence, Union, overload

import torch
from loguru import logger

# PathLike = TypeVar("PathLike", str, Path, os.PathLike)
from molreactgen.config import PathLike


# TODO Refactor Counter class
class Counter:
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
            raise AttributeError(f"Counter {counter} not found")

    @overload
    def get_count(self) -> dict[str, int]:
        ...

    @overload
    def get_count(self, counter: str) -> int:
        ...

    def get_count(self, counter: Optional[str] = None) -> Union[dict[str, int], int]:
        counts: Union[dict[str, int], int]
        if counter is None:
            counts = self._counters
        elif counter in self._counters:
            counts = self._counters[counter]
        else:
            raise AttributeError(f"Counter {counter} not found")

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
            raise AttributeError(f"Counter {counter} not found")

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
            raise AttributeError(f"Counter {counter} not found")

        return fractions


def show_warning(message: str, *_: Any, **__: Any) -> None:
    logger.warning(message)


def configure_logging(
    *,
    log_mode: str = "DEVELOPMENT",
    log_dir: Optional[PathLike] = None,
    log_file: Optional[PathLike] = None,
    console_log_level: Optional[str] = None,
    file_log_level: Optional[str] = None,
) -> None:

    # logger.debug(f'Module name: {__name__}')

    log_mode = log_mode.upper()

    # TODO test the log_modes

    if log_mode == "PRODUCTION":
        logger.disable("molgen")
        return

    elif log_mode == "EXPERIMENT":
        console_log_level = (
            "INFO" if console_log_level is None else console_log_level
        )
        file_log_level = "DEBUG" if file_log_level is None else file_log_level

    elif log_mode == "TEST":
        console_log_level = (
            "WARNING" if console_log_level is None else console_log_level
        )
        file_log_level = "DEBUG" if file_log_level is None else file_log_level

    elif log_mode == "DEVELOPMENT":
        console_log_level = (
            "DEBUG" if console_log_level is None else console_log_level
        )
        file_log_level = "DEBUG" if file_log_level is None else file_log_level

    else:
        raise ValueError(f"Invalid log_mode: {log_mode}")

    logger.remove()
    logger.level("HEADING", no=21, color="<red><BLACK><bold>")
    logger.add(sys.stderr, level=console_log_level)

    caller_file_path = Path(inspect.stack()[1].filename).resolve()
    if log_dir is None:
        log_dir = Path(caller_file_path.parent) / "logs"
    else:
        log_dir = Path(log_dir).resolve()
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = Path(str(caller_file_path.stem) + ".log")

    log_file = log_dir / log_file

    logger.add(
        sink=log_file,
        rotation="1 day",
        retention="7 days",
        level=file_log_level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        colorize=False,
    )

    warnings.showwarning = show_warning  # type: ignore

    logger.debug("Logging configured")
    logger.debug(f"Log to file '{log_file}'")


# TODO refactor name as get_hash_code
def get_hash_value(file_path: Union[str, Path]) -> int:
    file_path = Path(file_path).resolve()
    hash_fn = hashlib.md5()

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


# TODO is that used anywhere?
def create_file_link(
    link_file_path: PathLike, target_file_path: PathLike, hardlink: bool = False
) -> bool:
    link_file_path = Path(link_file_path)
    target_file_path = Path(target_file_path).resolve()
    link_file_path.unlink(missing_ok=True)
    if hardlink:
        # deprecated in 3.10, use link_file_path.hardlink_to(target_file_path) instead
        target_file_path.link_to(link_file_path)
    else:  # symlink
        link_file_path.symlink_to(target_file_path)

    return True


# Just replace with .resolve()?!
# def get_original_file_path(link_file_path: PathLike) -> Path:
#     link_file_path = Path(link_file_path)
#     if link_file_path.is_symlink():
#         return Path(link_file_path.readlink())
#     else:
#         return Path(link_file_path)


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


# TODO Refactor and test
"""
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
"""
