#!/usr/bin/env bash
# This script installs the required dependencies for the project.

CMAKE_ARGS="-DSD_VULKAN=ON" pip install -r requirements.txt --upgrade --force-reinstall