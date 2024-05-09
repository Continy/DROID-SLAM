#!/bin/bash


TARTANAIR_PATH=/datasets

python evaluation_scripts/validate_tartanair_v2_test.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis $@

