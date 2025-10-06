import re
import argparse

power_log_file = '../../log_power'

# Extract power from log_power file

with open(power_log_file, 'r') as f:
    all_lines = f.readlines()

    total_router_dynamic_power = 0
    total_router_leakage_power = 0

    total_link_dynamic_power = 0
    total_link_leakage_power = 0

    for line in range(len(all_lines)):
        if 'Total Power:' in all_lines[line]:
            # next line contains dynamic power : '    Dynamic power (W): 0.000795182'
            # next line contains leakage power : '    Leakage power (W): 8.72836e-05'
            dynamic_power = float(re.findall(r'[-+]?\d*\.?\d+(?:[ee][-+]?\d+)?', all_lines[line+1])[0])
            leakage_power = float(re.findall(r'[-+]?\d*\.?\d+(?:[ee][-+]?\d+)?', all_lines[line+2])[0])

            total_router_dynamic_power += dynamic_power
            total_router_leakage_power += leakage_power

        elif 'Link:' in all_lines[line]:
            # next line contains dynamic power : '    Dynamic power (W): 0.000000'
            # next line contains leakage power : '    Leakage power (W): 0.000000'
            dynamic_power = float(re.findall(r'[-+]?\d*\.?\d+(?:[ee][-+]?\d+)?', all_lines[line+2])[0])
            leakage_power = float(re.findall(r'[-+]?\d*\.?\d+(?:[ee][-+]?\d+)?', all_lines[line+3])[0])

            total_link_dynamic_power += dynamic_power
            total_link_leakage_power += leakage_power
        else:
            continue


argparser = argparse.ArgumentParser()
# add filename argument
argparser.add_argument('-f', '--filename', type=str, required=True)

args = argparser.parse_args()
filename = args.filename

with open(filename, 'w') as f:
    f.write(f'Total Router Dynamic Power: {total_router_dynamic_power}\n')
    f.write(f'Total Router Leakage Power: {total_router_leakage_power}\n')
    f.write(f'Total Link Dynamic Power: {total_link_dynamic_power}\n')
    f.write(f'Total Link Leakage Power: {total_link_leakage_power}\n')
    f.write(f'Total Power: {total_router_dynamic_power + total_router_leakage_power + total_link_dynamic_power + total_link_leakage_power}\n')
    