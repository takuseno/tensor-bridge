#!/bin/bash

pip install -e .
python setup.py build_ext --inplace
python scripts/benchmark.py
