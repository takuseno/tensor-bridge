#!/bin/bash

pip install -e .
pip install pytest
python -m pytest tests
