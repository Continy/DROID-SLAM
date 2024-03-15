#!/bin/bash


TARTANAIR_PATH=/compute/zoidberg/tartanair_v2

python evaluation_scripts/validate_tartanair_v2.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis $@

