# ImageEditor

A terminal-based modular image processing pipeline via JSON configs and custom filters.

## Features

- Supports blur, sharpen, brightness, contrast, and more
- Fully extendable — define your own filters by inheriting from BaseFilter
- Runs filters sequentially using config files
- CLI interface: `edit-image`

## Installation

```bash
pip install -e .
```

Or use the following script:

- install.sh

May need xdg-utils for displaying images on linux (+ wslview if on wsl)
