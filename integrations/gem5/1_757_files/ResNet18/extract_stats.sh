#!/bin/bash

bits=24

network_stats_file=network_stats_ResNet18_bits_"$bits".txt
power_stats_file=power_stats_ResNet18_bits_"$bits".txt

so extract_network_stats.sh $network_stats_file 
python extract_power.py '-f' $power_stats_file