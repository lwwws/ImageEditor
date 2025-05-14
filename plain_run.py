"""
plain_run.py
------
Running the pipeline with simple script, for testing purposes.
"""

from edit_image.pipeline import run_from_config

def main():
    config_path = r'configs/emboss.json'
    run_from_config(config_path, verbose=True)

if __name__ == '__main__':
    main()