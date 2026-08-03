"""
utils.py
Utility functions for image generation, including dimension calculations based on resolution and aspect ratio.
"""

import math


def calculate_dimensions(resolution: str, aspect_ratio: str):
    # 1. Determine base target area (1K defaults to 1024x1024)
    base_dim = 1024
    if resolution == "2K":
        base_dim = 2048
    elif resolution == "4K":
        base_dim = 4096

    target_area = base_dim * base_dim

    # 2. Parse the string aspect ratio (e.g., "16:9")
    w_ratio, h_ratio = map(float, aspect_ratio.split(":"))
    ratio = w_ratio / h_ratio

    # 3. Compute dimensions maintaining total pixel count
    ideal_height = math.sqrt(target_area / ratio)
    ideal_width = ideal_height * ratio

    # 4. Stable Diffusion requires dimensions to be multiples of 8
    width = int(round(ideal_width / 8.0) * 8)
    height = int(round(ideal_height / 8.0) * 8)

    return width, height
