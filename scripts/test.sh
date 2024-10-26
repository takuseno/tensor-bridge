#!/bin/bash

pip install -e .
python setup.py build_ext --inplace
pip install pytest
python -m pytest tests
