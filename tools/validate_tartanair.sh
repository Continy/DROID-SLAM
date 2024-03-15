#!/bin/bash


TARTANAIR_PATH=/data2/datasets/wenshanw/tartan_data

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis $@

