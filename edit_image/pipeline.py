"""
pipeline.py
-----------
Filter execution engine on images using a JSON-based configuration.

This script loads an image, parses a JSON pipeline of filters, dynamically imports
filter classes from `filters.catalog`, validates parameters, and sequentially applies
the filters to the image.

Usage:
    python pipeline.py path/to/config.json

Expected JSON format:
{
    "input": "input.jpg",
    "output": "output.jpg",         # optional
    "display": true,                # optional
    "operations": [
        {"Blur": {"strength": 0.4}},
        {"Sharpen": {"alpha": 1.5}}
    ]
}

ChatGPT Usage:
I've used chatgpt in order to generate most of the input validation checks.
I'm not very familiar with modules such as inspect, or importlib, so it helped me implement that too.
"""


import json
import importlib
import inspect
from PIL import Image
import numpy as np
import os
import time

# JSON key names
INPUT = 'input'
OUTPUT = 'output'
OPERATIONS = 'operations'
DISPLAY = 'display'
TYPE = 'type'

REQUIRED_KEYS = {INPUT, OPERATIONS}
AT_LEAST_ONE = {OUTPUT, DISPLAY}

# Path to ready to use filters
FILTERS_PATH = 'filters.catalog'

def log(msg, verbose=False):
    """
    Print a message if verbose mode is True.
    """
    if verbose:
        print(msg)

def validate_filter_config(filter_type, params, verbose=False):
    """
    Dynamically import and validate a filter class and its parameters.

    Parameters:
        filter_type (str): Name of the filter class/module (e.g. "Blur")
        params (dict): Arguments to pass to the filter's constructor
        verbose (bool): Enable logging

    Returns:
        type: The filter class

    Raises:
        ImportError, ValueError if module/class/params are invalid
    """

    try:
        module = importlib.import_module(f"{FILTERS_PATH}.{filter_type}")
    except ModuleNotFoundError:
        raise ImportError(f"Filter module '{FILTERS_PATH}.{filter_type}' not found")

    # Class name is assumed to be capitalized ...
    class_name = filter_type.capitalize()

    try:
        filter_class = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"Filter class '{filter_type}' not found in module")

    sig = inspect.signature(filter_class.__init__)
    valid_keys = set(sig.parameters.keys())
    invalid_keys = set(params) - valid_keys

    if invalid_keys:
        raise ValueError(f"Invalid parameters for filter '{filter_type}': {invalid_keys}")

    log(f"Validated filter: {filter_type} with params: {params}", verbose)
    return filter_class

def load_and_validate_config(config_path: str, verbose=False):
    """
    Load and validate a JSON pipeline config file.

    Parameters:
        config_path (str): Path to JSON config
        verbose (bool): Enable logging

    Returns:
        tuple: (config dict, list of (filter class, params) tuples)

    Raises:
        FileNotFoundError, ValueError, TypeError
    """

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")

    with open(config_path, "r") as f:
        config = json.load(f)

    if not REQUIRED_KEYS.issubset(set(config)):
        raise ValueError(f"Missing required config fields: {REQUIRED_KEYS - set(config)}")
    if not (AT_LEAST_ONE & set(config)):
        raise ValueError(f"Missing one of the following fields: {AT_LEAST_ONE}")

    if not os.path.exists(config[INPUT]):
        raise FileNotFoundError(f"Input image '{config[INPUT]}' not found")

    if not isinstance(config[OPERATIONS], list):
        raise TypeError(f"'{OPERATIONS} must be a list")

    # Validate all filters
    validated_filters = []

    for operation in config[OPERATIONS]:
        if not isinstance(operation, dict) or TYPE not in operation:
            raise ValueError(f"Each operation must be a dict with a '{TYPE}' field")

        op_type = operation[TYPE]
        op_params = {k: v for k, v in operation.items() if k != TYPE}

        cls = validate_filter_config(op_type, op_params, verbose)
        validated_filters.append((cls, op_params))

    return config, validated_filters

def resolve_output_path(path):
    """
    Return a safe output path. If the file already exists, appends _1, _2, ...

    Parameters:
        path (str): Desired path to output file

    Returns:
        str: non-conflicting output path
    """
    if not os.path.exists(path):
        return path

    log(f'File already exists! {path}', verbose=True)

    base, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{base}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1

def run_from_config(config_path: str, verbose=False):
    """
    Execute the full image filter pipeline from a config file.
    Opens, transforms, saves and/or displays an image.

    Parameters:
        config_path (str): Path to config file
        verbose (bool): Enable logging
    """

    config, filters = load_and_validate_config(config_path, verbose)

    log(f"Loading image: {config[INPUT]}", verbose)
    image = Image.open(config[INPUT]).convert("RGB")
    image_np = np.array(image)

    for idx, (cls, params) in enumerate(filters):
        log(f"Applying filter {idx + 1}: {cls.__name__} with params: {params}", verbose)
        filt = cls(**params)

        t0 = time.time()
        image_np = filt.apply_filter(image_np)
        log(f"filter took {time.time() - t0:.3f} seconds!", verbose)

    pil_image = Image.fromarray(image_np.astype(np.uint8))

    if OUTPUT in config:
        output_path = resolve_output_path(config[OUTPUT])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log(f"Saving output to: {output_path}", verbose)
        pil_image.save(output_path)

    if DISPLAY in config:
        pil_image.show()
