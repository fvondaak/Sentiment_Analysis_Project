#!/bin/sh

uv venv .venv
. .venv/bin/activate
uv pip install -r nbsvm_requirements.txt