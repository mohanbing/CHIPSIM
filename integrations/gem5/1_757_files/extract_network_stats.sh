#!/bin/bash

network_stats_file="$1"

echo > $network_stats_file
grep "simTicks" ../m5out/stats.txt | sed 's/simTicks\s*/simTicks = /' >> $network_stats_file
grep "average_packet_latency" ../m5out/stats.txt | sed 's/system.ruby.network.average_packet_latency\s*/average_packet_latency = /' >> $network_stats_file
grep "average_packet_queueing_latency" ../m5out/stats.txt | sed 's/system.ruby.network.average_packet_queueing_latency\s*/average_packet_queueing_latency = /' >> $network_stats_file
grep "average_packet_network_latency" ../m5out/stats.txt | sed 's/system.ruby.network.average_packet_network_latency\s*/average_packet_network_latency = /' >> $network_stats_file
grep "packets_injected::total" ../m5out/stats.txt | sed 's/system.ruby.network.packets_injected::total\s*/packets_injected = /' >> $network_stats_file
grep "packets_received::total" ../m5out/stats.txt | sed 's/system.ruby.network.packets_received::total\s*/packets_received = /' >> $network_stats_file
grep "average_hops" ../m5out/stats.txt | sed 's/system.ruby.network.average_hops\s*/average_hops = /' >> $network_stats_file