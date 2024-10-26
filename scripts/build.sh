#!/bin/bash

# clean
rm -rf tensor_bridge.egg-info
rm -rf dist
rm -rf build

python setup.py sdist bdist_wheel
