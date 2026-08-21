"""
config.py
Configuration constants and cost tables for the Nano Banana & BytePlus Seedream APIs.
"""

import os

# --- Cloud Models (Google GenAI) ---
# Available models based on the Nano Banana family
GEMINI_IMAGE_MODELS = [
    "gemini-3.1-flash-lite-image",  # Nano Banana 2 Lite
    "gemini-3.1-flash-image",  # Nano Banana 2
    "gemini-3-pro-image",  # Nano Banana Pro
    "gemini-2.5-flash-image",  # Legacy
]

# --- Cloud Models (BytePlus Seedream) ---
SEEDREAM_MODELS = [
    "seedream-5-0-pro",
    "seedream-5-0-lite",
    "seedream-4-5",
    "seedream-4-0",
]

RESOLUTIONS = ["1K", "2K", "4K"]
ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "2:3", "3:2"]

# Estimated cost table in integer cents (1 unit = $0.01)
# Includes Nano Banana baseline estimates and BytePlus per-image pricing.
COST_TABLE_CENTS = {
    # Google Models
    "gemini-3.1-flash-lite-image": {"1K": 1, "2K": 2, "4K": 4},
    "gemini-3.1-flash-image": {"1K": 3, "2K": 6, "4K": 12},
    "gemini-3-pro-image": {"1K": 6, "2K": 12, "4K": 24},
    "gemini-2.5-flash-image": {"1K": 2, "2K": 4, "4K": 8},
    # BytePlus Seedream Models (Flatter per-image rates)
    "seedream-5-0-pro": {"1K": 12, "2K": 12, "4K": 12},
    "seedream-5-0-lite": {"1K": 4, "2K": 4, "4K": 4},
    "seedream-4-5": {"1K": 4, "2K": 4, "4K": 4},
    "seedream-4-0": {"1K": 3, "2K": 3, "4K": 3},
}

# Estimated cost table for Google Cloud Batch API processing (50% discount applied)
BATCH_COST_TABLE_CENTS = {
    "gemini-3.1-flash-lite-image": {"1K": 1, "2K": 1, "4K": 2},
    "gemini-3.1-flash-image": {"1K": 2, "2K": 3, "4K": 6},
    "gemini-3-pro-image": {"1K": 3, "2K": 6, "4K": 12},
    "gemini-2.5-flash-image": {"1K": 1, "2K": 2, "4K": 4},
}

# --- Local Models ---
LOCAL_MODELS_DIR = "models"
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)


def get_local_models():
    """Scans the local directory for GGUF and SafeTensors files."""
    if not os.path.exists(LOCAL_MODELS_DIR):
        return []
    return [f for f in os.listdir(LOCAL_MODELS_DIR) if f.endswith((".safetensors", ".gguf"))]


# --- Settings ---
SETTINGS_FILE = "settings.json"

# --- Database ---
DB_PATH = "nano_banana_cache.db"

# --- API status messages ---
API_STATUS_MESSAGES = {
    "gemini": {
        "success": "🟢 **Gemini API Status:** Connected & Reachable",
        "failure": "🔴 **Gemini API Status:** Disconnected / Invalid Google Credentials",
        "default": "⚪ **Gemini API Status:** Waiting for credentials (API key or project ID)...",
    },
    "byteplus": {
        "success": "🟢 **BytePlus API Status:** Connected & Reachable",
        "failure": "🔴 **BytePlus API Status:** Disconnected / Invalid BytePlus Credentials",
        "default": "⚪ **BytePlus API Status:** Waiting for API key...",
    },
}
