# coding=utf-8
# src/molreactgen/helpers.py
"""General helper functions.

Classes:
    _ArgsEncoder:
        A JSON encoder for arguments of type Path.
    Tally:
        A simple tally class for counting things.

Functions:
    determine_log_level:
        Determine the log level from command line arguments.
    configure_logging:
        Configure loguru logging.
    guess_project_root_dir:
        Guess the project´s root directory.
    get_hash_code:
        Get a hash code for a given file or directory of files.
    save_commandline_arguments:
        Save the command line arguments to a JSON file.
    create_file_link:
        Create a link to a file.
    get_num_workers:
        Get a default number of workers for multiprocessing.
    get_device:
        Get the torch device to use for computations (single gpu case only).
    get_device_type:
        Get basic information about the device type (single gpu case only).
"""


import argparse
import contextlib
import hashlib
import inspect
import json
import logging
import os
import sys
import warnings
from functools import partialmethod
from logging.handlers import SysLogHandler
from os import PathLike
from pathlib import Path, PosixPath
from types import FrameType
from typing import Any, Final, Generator, Iterable, Optional, Sequence, Union, overload

import torch
from loguru import logger

LOG_LEVELS: Final[tuple[int, ...]] = (
    logging.DEBUG,
    logging.INFO,
    logging.WARNING,
    logging.ERROR,
    logging.CRITICAL,
)

# This is the default format used by loguru
# DEFAULT_CONSOLE_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | " \
#                          "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " \
#                          "<level>{message}</level>"
DEFAULT_CONSOLE_FORMAT: Final[
    str
] = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
DEFAULT_FILE_FORMAT: Final[str] = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | "
    "{name}:{function}:{line} - <level>{message}</level>"
)
DEFAULT_SYSLOG_FORMAT: Final[str] = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | "
    "{file}:{name}:{function}:{line} - <level>{message}</level>"
)

DEFAULT_LOG_LEVEL: Final[int] = logging.INFO
SIGNS_FOR_ROOT_DIR: Final[tuple[str, ...]] = (
    ".git",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
)


###############################################################################
# Tally, used during generation                                             #
###############################################################################


class Tally:
    """A simple tally class for counting things.

    The counters have a hierarchical structure with the hierarchy defined by the order of the counter names.
    The base counter is always the first counter.
    Absolute fractions are calculated with respect to the base counter.
    Relative fractions are calculated with respect to the counter one hierarchy level up.

    Methods:
        increment: increment a counter by a given amount
        get_count: get the current count of a given or all counter(s)
        get_absolute_fraction: get the absolute fraction of a given or all counter(s)
        get_relative_fraction: get the relative fraction of a given or all counter(s)
        save_to_file: save the counters to a file
    """

    _base_name = "__base__"
    _valid_file_formats: dict[str, tuple[str, ...]] = {
        # "CSV": (".CSV",),
        "JSON": (".JSON",),
    }

    def __init__(self, counters: Union[Sequence[str], str]) -> None:
        """Initialize the Tally object.

        Args:
            counters: a list of counter names or a single counter name
        """

        # Validate arguments
        if isinstance(counters, str) or not isinstance(counters, Iterable):
            counters = [counters]

        counters = [str(c) for c in counters]
        if self._base_name in counters:
            raise ValueError(f"counter name {self._base_name} is reserved")

        if len(counters) != len(set(counters)):
            raise ValueError("counters must be unique")

        # Initialize counters, setup counter hierarchy
        self._counters: dict[str, int] = {str(k): 0 for k in counters}
        self._key_to_idx: dict[str, int] = {str(k): i for i, k in enumerate(counters)}
        self._idx_to_key: dict[int, str] = {i: str(k) for i, k in enumerate(counters)}
        counters_one_up = [self._base_name]
        counters_one_up.extend(counters[:-1])
        assert len(counters) == len(counters_one_up)

    def _get_value_from_idx(self, idx: int) -> int:
        return self._counters[self._idx_to_key[idx]]

    def increment(self, counter: str, increment: int = 1) -> None:
        """Increment a counter by a given amount.

        Args:
            counter: the counter to increment.
            increment: the amount to increment the counter by, defaults to 1.
        """

        if counter in self._counters:
            self._counters[counter] += increment
        else:
            raise AttributeError(f"Counter {counter} not found")

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_count(
        self,
        counter: Optional[Any] = None,
        *,
        format_specifier: Optional[str] = None,
    ) -> Union[dict[str, int], dict[str, str]]:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_count(
        self, counter: Sequence[str], *, format_specifier: Optional[str] = None
    ) -> Union[dict[str, int], dict[str, str], int, str]:
        ...

    def get_count(
        self,
        counter: Optional[Union[Sequence[str], str]] = None,
        *,
        format_specifier: Optional[str] = None,
    ) -> Union[dict[str, int], dict[str, str], int, str]:
        """Get the current count of (a) given counter(s) or all counters.

        Args:
            counter: the counter(s) to get the count of, defaults to None, i.e. all counters.
            format_specifier: a format specifier for the count, defaults to None.

        Returns:
            the count(s) of the counter(s) as a dictionary or a single value.
        """

        counts: Union[dict[str, int], int]

        # Get counts
        if counter is None:  # defaults to all counters
            counts = self._counters

        elif not isinstance(counter, str) and isinstance(counter, Iterable):
            for k in counter:
                if k not in self._counters:
                    raise AttributeError(f"Counter {k} not found")
            counts = {k: self._counters[k] for k in counter}

        elif counter in self._counters:
            counts = self._counters[counter]

        else:
            raise AttributeError(f"Counter {counter} not found")

        # Format counts
        if format_specifier is not None and isinstance(format_specifier, str):
            return (
                {k: format(v, format_specifier) for k, v in counts.items()}
                if isinstance(counts, dict)
                else format(counts, format_specifier)
            )
        else:
            return counts

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_absolute_fraction(
        self,
        counter: Optional[Any] = None,
        *,
        format_specifier: Optional[str] = None,
    ) -> Union[dict[str, float], dict[str, str]]:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_absolute_fraction(
        self, counter: Sequence[str], *, format_specifier: Optional[str] = None
    ) -> Union[dict[str, float], dict[str, str], float, str]:
        ...

    def get_absolute_fraction(
        self,
        counter: Optional[Union[Sequence[str], str]] = None,
        *,
        format_specifier: Optional[str] = None,
    ) -> Union[dict[str, float], dict[str, str], float, str]:
        """Get the absolute fraction of (a) given counter(s) or all counters.

        Absolute refers to the fraction of the counter relative to the base counter.
        The base counter is the first counter in the sequence given to the constructor.

        Args:
            counter: the counter(s) to get the absolute fraction of, defaults to None, i.e. all counters.
            format_specifier: a format specifier for the absolute fraction, defaults to None.

        Returns:
            the absolute fraction(s) of the counter(s) as a dictionary or a single value.
        """

        # Get absolute fractions
        counts = self.get_count(counter)
        base_value = self._get_value_from_idx(0)
        fractions: Union[dict[str, float], float]

        if isinstance(counts, dict):  # counter was/is None
            if base_value == 0:
                fractions = {k: float("nan") for k in counts}
            else:
                fractions = {k: int(v) / base_value for k, v in counts.items()}

        elif isinstance(counter, str) and counter in self._counters:
            if base_value == 0:
                fractions = float("nan")
            else:
                fractions = self._counters[counter] / base_value

        else:
            raise AttributeError(f"Counter {counter} not found")

        # Format absolute fractions
        if format_specifier is not None and isinstance(format_specifier, str):
            return (
                {k: format(v, format_specifier) for k, v in fractions.items()}
                if isinstance(fractions, dict)
                else format(fractions, format_specifier)
            )
        else:
            return fractions

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_relative_fraction(
        self, *, format_specifier: Optional[str] = None
    ) -> Union[dict[str, float], dict[str, str]]:
        ...

    # noinspection PyMissingOrEmptyDocstring
    @overload
    def get_relative_fraction(
        self, counter: str, *, format_specifier: Optional[str] = None
    ) -> Union[float, str]:
        ...

    def get_relative_fraction(
        self,
        counter: Optional[str] = None,
        *,
        format_specifier: Optional[str] = None,
    ) -> Union[dict[str, float], dict[str, str], float, str]:
        """Get the relative fraction of (a) given counter(s) or all counters.

        Relative refers to the fraction of the counter relative to the counter one hierarchy up.
        The hierarchy is defined by the sequence of names given to the constructor.

        Args:
            counter: the counter(s) to get the relative fraction of, defaults to None, i.e. all counters.
            format_specifier: a format specifier for the relative fraction, defaults to None.

        Returns:
            the relative fraction(s) of the counter(s) as a dictionary or a single value.
        """

        counts = self.get_count()
        count_names = list(counts.keys())
        count_values = list(counts.values())
        count_names_one_up = [self._base_name]
        count_names_one_up.extend(count_names[-1])
        count_values_one_up = [count_values[0]]
        count_values_one_up.extend(count_values[:-1])
        # assert isinstance(counts, dict) and all(
        #     isinstance(v, int) for v in counts.values()
        # )
        fractions: Union[dict[str, float], float]
        if counter is None or (
            not isinstance(counter, str) and isinstance(counter, Iterable)
        ):
            # mypy seems to determine "object" as the common base class for c / count_values
            fraction_values = [
                c / c_one_up if c_one_up != 0 else float("nan")  # type: ignore
                for c, c_one_up in zip(count_values, count_values_one_up)
            ]
            fractions = dict(zip(count_names, fraction_values))
            if not isinstance(counter, str) and isinstance(counter, Iterable):
                fractions = {k: v for k, v in fractions.items() if k in counter}

        elif counter in self._counters:
            idx = self._key_to_idx[counter]
            if count_values_one_up[idx] == 0:
                fractions = float("nan")
            else:
                # mypy seems to determine "object" as the common base class for c / count_values
                fractions = count_values[idx] / count_values_one_up[idx]  # type: ignore
        else:
            raise AttributeError(f"Counter {counter} not found")

        if format_specifier is not None and isinstance(format_specifier, str):
            return (
                {k: format(v, format_specifier) for k, v in fractions.items()}
                if isinstance(fractions, dict)
                else format(fractions, format_specifier)
            )
        else:
            return fractions

    # might implement this if needed
    # def _save_to_csv(self, file_path: Path) -> None:
    #     pass

    def _save_to_json(self, file_path: Path) -> None:
        dict_to_save = {
            "counts": self.get_count(),
            "absolute_fractions": self.get_absolute_fraction(),
            "relative_fractions": self.get_relative_fraction(),
        }
        with open(file_path, "w") as f:
            json.dump(dict_to_save, f)

    def save_to_file(
        self, file_path: Union[PathLike, str], format_: str = "json"
    ) -> None:
        """Save the counters to a file.

        Currently only JSON format is supported.

        Args:
            file_path: the path to the file to save to.
            format_: the format to save the file in, defaults to "json".
        """

        file_path = Path(file_path).resolve()
        if format_.upper() not in self._valid_file_formats:
            raise ValueError(f"Unknown format {format_}")

        file_extension = file_path.suffix
        if file_extension.upper() not in self._valid_file_formats[format_.upper()]:
            logger.warning(
                f"File format {format_} does not match file extension {file_extension}, saving anyway"
            )

        if format_.upper() == "JSON":
            self._save_to_json(file_path)
        else:
            raise ValueError(f"Unknown format {format_}")

    def __repr__(self) -> str:
        class_name = type(self).__name__
        counters = list(self._counters.keys())
        return f"{class_name}" f"({counters})"

    def __str__(self) -> str:
        return str([f"{k}: {v}" for k, v in self._counters.items()])


###############################################################################
# Loguru Logging                                                              #
###############################################################################


def determine_log_level(
    adjustments: Optional[Iterable[int]] = None,
    default_log_level: int = DEFAULT_LOG_LEVEL,
    log_levels: Iterable[int] = LOG_LEVELS,
) -> int:
    """Determine the log level based on the given adjustments and default log level.

    The adjustments are given as a sequence of integers.
    Usually -1 or 1 corresponding to command line arguments such as -q (quiet) and -v (verbose).

    Args:
        adjustments: the adjustments to the log level, defaults to None, i.e. no adjustments.
        default_log_level: the default log level, defaults to DEFAULT_LOG_LEVEL.
        log_levels: the log levels to choose from, defaults to LOG_LEVELS.

    Returns:
        the log level to use.
    """

    adjustments = () if adjustments is None else adjustments
    log_levels = sorted(list(log_levels))
    log_level_idx: int = log_levels.index(default_log_level)
    # For each adjustment ("-q" and "-v" flag), adjust the logging verbosity accordingly
    # making sure to clamp off the value from 0 to the length of the log levels, inclusive of both
    for adj in adjustments:
        log_level_idx = min(len(log_levels) - 1, max(log_level_idx + adj, 0))

    return log_levels[log_level_idx]


# Code taken from https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
# Type annotations are my own
class _InterceptHandler(logging.Handler):
    """Intercept standard logging messages and forward them to loguru."""

    # noinspection PyMissingOrEmptyDocstring
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
        # noinspection PyProtectedMember
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _show_warning(message: str, *_: Any, **__: Any) -> None:
    """Redirect warnings to loguru."""
    logger.warning(message)


def configure_logging(
    log_level: int = logging.INFO,
    *,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    syslog_format: Optional[str] = None,
    file_log_level: Optional[int] = None,
    log_dir: Optional[Union[PathLike, str]] = None,
    log_file: Optional[Union[PathLike, str]] = None,
    rotation: str = "1 day",
    retention: str = "7 days",
    address: Optional[tuple[str, int]] = None,
) -> None:
    """Configure logging.

    Args:
        log_level: the log level to use, defaults to logging.INFO.
        console_format: the format to use for the console, defaults to None, i.e. use the default format.
        file_format: the format to use for the file, defaults to None, i.e. use the default format.
        syslog_format: the format to use for syslog, defaults to None, i.e. use the default format.
        file_log_level: the log level to use for the file, defaults to None, i.e. use the console log level.
        log_dir: the directory to use for the log file, defaults to None, i.e. use <project root>/logs.
        log_file: the name of the log file, defaults to None, i.e. use <caller name>.log.
        rotation: the rotation to use for the log file, defaults to "1 day".
        retention: the retention to use for the log file, defaults to "7 days".
        address: the address to use for syslog, defaults to None, i.e. do not use syslog.
    """

    # Allow for separate log levels for console and file logging
    console_log_level = log_level
    file_log_level = console_log_level if file_log_level is None else file_log_level
    syslog_log_level = file_log_level

    console_format = (
        DEFAULT_CONSOLE_FORMAT if console_format is None else str(console_format)
    )
    file_format = DEFAULT_FILE_FORMAT if file_format is None else str(file_format)
    syslog_format = (
        DEFAULT_SYSLOG_FORMAT if syslog_format is None else str(syslog_format)
    )

    # Determine log file path
    # To be changed with python version ≥ 3.11: A list of FrameInfo objects is returned.
    caller_file_path = Path(inspect.stack()[1].filename).resolve()
    if log_dir is None:
        log_dir = guess_project_root_dir(caller_file_path=caller_file_path) / "logs"
    else:
        log_dir = Path(log_dir).resolve()
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = Path(f"{str(caller_file_path.stem)}.log")

    log_file = log_dir / log_file

    # Remove all previously added handlers
    logger.remove()

    # Allow for logging of headers, add log level
    with contextlib.suppress(TypeError):
        logger.level("HEADING", no=21, color="<red><BLACK><bold>")
        logger.__class__.heading = partialmethod(logger.__class__.log, "HEADING")  # type: ignore

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

    # Add syslog (papertrail) handler, experimental
    if address is not None:
        syslog_handler = SysLogHandler(address=address)
        logger.add(syslog_handler, level=syslog_log_level, format=syslog_format)

    # Redirect warnings to logger
    warnings.showwarning = _show_warning  # type: ignore

    # Intercept logging messages from other libraries
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    logger.debug(
        f"Logging with log level {file_log_level} to {log_file} with rotation {rotation} and retention {retention}"
    )


def configure_hf_logging(log_level: int = logging.INFO) -> None:
    """Configure logging for Hugging Face libraries.

    Args:
        log_level: the log level to use, defaults to logging.INFO.
    """

    if "datasets" in sys.modules:
        logger.debug("Configuring Hugging Face datasets logging...")
        import datasets  # type: ignore

        datasets.utils.logging.set_verbosity(log_level)
        datasets.utils.logging.enable_progress_bar()
        datasets.utils.logging.disable_propagation()

    if "evaluate" in sys.modules:
        logger.debug("Configuring Hugging Face evaluate logging...")
        import evaluate  # type: ignore

        evaluate.utils.logging.set_verbosity(log_level)
        evaluate.utils.logging.enable_progress_bar()
        evaluate.utils.logging.disable_propagation()

    if "transformers" in sys.modules:
        logger.debug("Configuring Hugging Face transformers logging...")
        import transformers  # type: ignore

        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.disable_default_handler()
        # transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.enable_progress_bar()


###############################################################################
# File related                                                                #
###############################################################################


def guess_project_root_dir(
    caller_file_path: Optional[Union[PathLike, str]] = None,
    signs_for_root_dir: Iterable[str] = SIGNS_FOR_ROOT_DIR,
) -> Path:
    """Guess the project´s root directory.

    Traverses the directory from the caller´s directory toward the root directory.
    The first directory that contains one of the signs for the root directory becomes the root directory.
    If no such directory is found, the caller´s directory is returned.

    Args:
        caller_file_path: the path to the caller´s file, defaults to None, i.e. use the caller´s file.
        signs_for_root_dir: the signs for the root directory, defaults to SIGNS_FOR_ROOT_DIR.

    Returns:
        The project´s root directory.
    """

    caller_file_path = (
        Path(inspect.stack()[1].filename).resolve()
        if caller_file_path is None
        else Path(caller_file_path).resolve()
    )
    directory = caller_file_path
    while (directory != directory.parent) and (directory := directory.parent):
        if any(directory.joinpath(sign).exists() for sign in signs_for_root_dir):
            return directory

    return caller_file_path.parent


def get_hash_code(file_path: Union[str, Path], algorithm: str = "sha256") -> int:
    """Get the hash code of a file or directory.

    Args:
        file_path: the path to the file or directory.
        algorithm: the hash algorithm, defaults to "sha256".

    Returns:
        The hash code.
    """

    # Validate arguments
    file_path = Path(file_path).resolve()
    if algorithm.upper() == "MD5":
        hash_fn = hashlib.md5()
    elif algorithm.upper() == "SHA256":
        hash_fn = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm '{algorithm}'")

    # Build file list
    file_list: Union[list[Path], Generator[Path, None, None]]
    if file_path.is_file():
        file_list = [file_path]
    elif file_path.is_dir():
        file_list = file_path.rglob("*")
    else:
        raise ValueError(f"Invalid file_path: {file_path}")

    # Calculate hash code
    for file in file_list:
        if file.is_file():
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(hash_fn.block_size), b""):
                    hash_fn.update(chunk)

    return int(hash_fn.hexdigest(), 16)


class _ArgsEncoder(json.JSONEncoder):
    """JSON encoder for PosixPath (as used by pathlib)."""

    # noinspection PyMissingOrEmptyDocstring
    def default(self, x: Any) -> Any:
        return x.as_posix() if isinstance(x, PosixPath) else super().default(x)


def save_commandline_arguments(
    args: argparse.Namespace,
    file_path: Union[PathLike, str],
    keys_to_remove: Optional[Iterable[str]] = None,
) -> None:
    """Save command-line arguments to a JSON file.

    Args:
        args: the command-line arguments.
        file_path: the path to the JSON file.
        keys_to_remove: the keys to remove from the arguments, defaults to None, i.e. remove nothing.
    """

    dict_to_save = dict(args.__dict__)
    if keys_to_remove is not None:
        for key in keys_to_remove:
            dict_to_save.pop(key, None)
    file_path = Path(file_path).resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Saving command-line arguments to {file_path}...")
    with open(file_path, "w") as f:
        json.dump(dict_to_save, f, cls=_ArgsEncoder, indent=4, sort_keys=True)


def create_file_link(
    from_file_path: Union[PathLike, str],
    to_file_path: Union[PathLike, str],
    hard_link: bool = True,
) -> None:
    """Create a link from a file to another file.

    Args:
        from_file_path: the path to the file to link from.
        to_file_path: the path to the file to link to.
        hard_link: whether to create a hard link, defaults to False, i.e. create a symlink (depends on OS behavior).
    """

    from_file_path = Path(from_file_path)
    to_file_path = Path(to_file_path).resolve()
    if not to_file_path.exists():
        raise FileNotFoundError(f"File {to_file_path} does not exist")
    from_file_path.unlink(missing_ok=True)

    if hard_link:
        # The link_to() method is deprecated in python v3.12
        # Use the hardlink_to() method instead (see below)
        # You can use the hardlink_to() method starting with python v3.10
        try:
            to_file_path.link_to(from_file_path)
            # from_file_path.hardlink_to(to_file_path)  # yes, this is the correct order
            return
        except PermissionError:
            # TODO investigate why this happens
            logger.warning("Failed to create a hard link, use symbolic link instead")

    from_file_path.symlink_to(to_file_path)  # symlink


###############################################################################
# Miscellaneous                                                               #
###############################################################################


def get_num_workers(spare_cpus: int = 1) -> int:
    """Determine a reasonable number of workers for multiprocessing.

    Args:
        spare_cpus: the number of CPUs to spare, defaults to 1.

    Returns:
        The number of workers.
    """

    spare_cpus = max(spare_cpus, 0)
    num_cpus = os.cpu_count()
    if num_cpus is None:
        num_cpus = 1

    return max(num_cpus - spare_cpus, 1)


def get_device() -> torch.device:
    """Determine available device.

    Returns:
        device(type='cuda') if cuda is available, device(type='cpu') otherwise.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_device_type() -> str:
    """Provides GPU type/name (for information only)

    Returns:
        GPU type/name
    """

    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
