"""
cli.py
------
Command-line interface for running image filter pipelines.

Usage:
    python cli.py --config path/to/config.json [--verbose]

ChatGPT Usage:
I used it to for argument parsers

Plus I used it to generate install.sh

Also, speaking in general, throughout the project I've used it for documentation purposes,
 finding bugs and general conventions for implementation
"""

import argparse
from edit_image.pipeline import run_from_config

def main():
    parser = argparse.ArgumentParser(
        description='Apply image filters defined in a JSON config file.'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to the JSON configuration file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed logging'
    )

    args = parser.parse_args()
    run_from_config(args.config, verbose=args.verbose)

if __name__ == '__main__':
    main()