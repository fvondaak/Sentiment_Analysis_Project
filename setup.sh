#!/usr/bin/env bash
# Run this to setup virtual environment with pip instead of uv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
