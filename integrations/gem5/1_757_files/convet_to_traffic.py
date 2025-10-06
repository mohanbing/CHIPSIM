import csv
import numpy as np


bits_drop = 22

# network = 'VGG19' # 17 layers
# network = 'ResNet18' # 21 layers
# network = 'PreActResNet18' # 18 layers
network = 'MobileNet' # 28 layers
# network = 'DenseNet121' # 120 layers
data_file = f'{network}_traffic.txt'

if network == 'VGG19':
    random_tiles = [13, 17, 7, 15, 16, 4, 1, 11, 10, 5, 3, 6, 8, 9, 2, 0, 14, 12] #vgg19
    total_nodes = 18 #6x3
elif network == 'ResNet18':
    random_tiles = [12, 3, 9, 17, 5, 10, 19, 16, 20, 2, 13, 11, 8, 1, 15, 7, 14, 6, 4, 0, 18] #resnet18
    total_nodes = 24 #6x4
elif network == 'PreActResNet18':
    random_tiles = [13, 0, 6, 16, 5, 9, 14, 2, 17, 8, 12, 15, 1, 7, 10, 3, 11, 4] #preactresnet18
    total_nodes = 18 #6x3
elif network == 'MobileNet':
    random_tiles = [9, 22, 13, 24, 18, 5, 23, 14, 8, 20, 1, 7, 11, 10, 15, 17, 12, 19, 27, 0, 2, 26, 25, 4, 6, 3, 21, 16] #mobilenet
    total_nodes = 30 #6x5 
elif network == 'DenseNet121':
    random_tiles = [28, 71, 19, 104, 43, 60, 6, 40, 72, 49, 116, 76, 83, 15, 77, 89, 50, 2, 88, 98, 94, 
                    108, 112, 31, 46, 55, 59, 106, 113, 85, 36, 79, 87, 10, 26, 17, 23, 102, 57, 44, 33, 
                    67, 84, 32, 74, 109, 64, 48, 12, 3, 39, 9, 69, 25, 62, 4, 86, 20, 14, 80, 21, 95, 
                    114, 118, 27, 61, 90, 75, 5, 92, 115, 7, 103, 99, 29, 101, 37, 78, 93, 68, 53, 70, 
                    111, 91, 52, 110, 30, 8, 41, 45, 51, 96, 65, 34, 11, 0, 18, 1, 22, 42, 81, 56, 105, 
                    54, 97, 58, 100, 117, 24, 47, 107, 82, 35, 38, 119, 66, 73, 63, 16, 13] #densenet121
    total_nodes = 120 #6x20

def convert_traffic_pattern(data):
    traffic_pattern = []
    max_cycles = 0
    for row in data:
        # each row: source_tile, dest_tile, traffic
        source_tile, dest_tile, traffic = row
        full_cycles, remaining_pkts = divmod(traffic*(32-bits_drop),128*8) # total packets to send, bits

        if remaining_pkts > 0:
            full_cycles += 1

        if full_cycles > max_cycles:
            max_cycles = full_cycles
        traffic_pattern.append((full_cycles, remaining_pkts, source_tile, dest_tile))

    cycle_traffic = []
    for cycle in range(max_cycles):
        tiles_this_cycle = []
        for row in traffic_pattern:
            tile_start = row[2]
            tile_end = row[3]
            vnet = 0
            if tile_start in tiles_this_cycle:
                vnet += 1
            tiles_this_cycle.append(tile_start)

            full_cycles = row[0] 
            cycles_left = full_cycles - cycle
            if cycles_left > 0:
                # traffic : cycle, i_ni, i_router, o_ni, o_router, vnet, flit
                if cycles_left == 1:
                    remaining_pkts = row[1]
                    if remaining_pkts > 0:
                        cycle_traffic.append((cycle*8+1, random_tiles[tile_start], random_tiles[tile_start], random_tiles[tile_end] + total_nodes, random_tiles[tile_end], vnet, remaining_pkts))
                    else:
                        cycle_traffic.append((cycle*8+1, random_tiles[tile_start], random_tiles[tile_start], random_tiles[tile_end] + total_nodes, random_tiles[tile_end], vnet, 8))
                else:
                    cycle_traffic.append((cycle*8+1, random_tiles[tile_start], random_tiles[tile_start], random_tiles[tile_end] + total_nodes, random_tiles[tile_end], vnet, 8))

    return cycle_traffic

# 1st row is header, convert the rest of data to integer
with open(data_file, 'r') as f:
    data = list(csv.reader(f))[1:]

data = [(int(row[0]), int(row[1]), int(row[2])) for row in data]
cycle_traffic = convert_traffic_pattern(data)
cycle_traffic = np.array(cycle_traffic)

garnet_traffic_file = f'{network}/{network}_bits_{bits_drop}_garnet_traffic.txt'

header = 'cycle i_ni i_router o_ni o_router vnet flit'

np.savetxt(garnet_traffic_file, cycle_traffic, fmt='%d', header=header, delimiter=' ')
