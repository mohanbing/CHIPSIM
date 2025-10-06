# Chiplet performance parameters (for different chiplet types)
CHIPLET_TYPES = [
    {
        'name': 'Standard',         # Basic Analog
        'tops': 3.06176,               # Terra operations per second
        'energy_per_mac': 873.109504,      # in fJ
        # 'idle_power_per_crossbar': 0.0001,  # in Watts (example value)
        # 'active_power_per_crossbar': 0.00296,    # in Watts, not used
        'area': 4.0,                # Area normalized to the size of the standard chiplet
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 2,         # Resolution of the individual crossbarl cells
        'dac_precision_bits': 1,    
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 299,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 330,      # in degree Kelvin
    },
    {
        'name': 'Raella',         # Raella
        'tops': 3.06176,               # Terra operations per second
        'energy_per_mac': 873.109504,      # in fJ
        # 'idle_power_per_crossbar': 0.0001,  # in Watts (example value)
        # 'active_power_per_crossbar': 0.00296,    # in Watts, not used
        'area': 4.0,                # Area normalized to the size of the standard chiplet
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 2,         # Resolution of the individual crossbarl cells
        'dac_precision_bits': 1,    
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 299,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 330,      # in degree Kelvin
    },
    { 
        'name': 'Shared_ADC',       # Macro A, jia_jssc_2020
        'tops': 3.13344,       
        'energy_per_mac': 303.928319999999,
        # 'idle_power_per_crossbar': 0.000808,
        # 'active_power_per_crossbar': 0.03248,
        'area': 120,
        'crossbar_rows': 768,
        'crossbar_columns': 768,
        'bits_per_cell': 1,
        'dac_precision_bits': 1,
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 17,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 358,      # in degree Kelvin
    },
    {
        'name': 'Adder',            # Macro B, sinangil_jssc_2021
        'tops': 1.10592,     
        'energy_per_mac': 177.2664832,     
        # 'idle_power_per_crossbar': 0.00000342,
        # 'active_power_per_crossbar': 0.02049, # not being used
        'area': 30,
        'crossbar_rows': 64,
        'crossbar_columns': 64,
        'bits_per_cell': 1,
        'dac_precision_bits': 4,
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 216,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 358,      # in degree Kelvin
    },
    {
        'name': 'Accumulator',      # Macro C, wan nature 2022
        'tops': 3.510857143,         # Adjusted value
        'energy_per_mac': 222.45888,
        # 'idle_power_per_crossbar': 0.0001,  # in Watts (example value)
        # 'active_power_per_crossbar': 0.0005939,
        'area': 500,
        'crossbar_rows': 256,
        'crossbar_columns': 256,
        'bits_per_cell': 2,
        'dac_precision_bits': 1,
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 150,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 330,      # in degree Kelvin
    },
    {
        'name': 'ADC_less',         # Macro E, colonnade jssc 2021
        'tops': 0.38656,         # Adjusted value
        'energy_per_mac': 268.704,
        # 'idle_power_per_crossbar': 0.00122,  # in Watts (example value)
        # 'active_power_per_crossbar': 0.03727,
        'area': 600,
        'crossbar_rows': 128,
        'crossbar_columns': 128,
        'bits_per_cell': 1,
        'dac_precision_bits': 1,
        'bits_per_weight': 8,
        'input_precision_bits': 8,
        'tiles_per_chiplet': 1,
        'crossbars_per_tile': 151,
        # 'threshold_temp' : 450,      # in degree Kelvin
        'threshold_temp' : 358,      # in degree Kelvin
    },
    {
        'name': 'IO',               # I/O chiplet for weight loading
        'tops': 0,                  # I/O chiplets don't perform compute
        'energy_per_mac': 0,        # No MAC operations
        'area': 50,                 # Made up value for I/O chiplet area
        'crossbar_rows': 0,         # No crossbars
        'crossbar_columns': 0,      # No crossbars
        'bits_per_cell': 0,         # No memory cells
        'dac_precision_bits': 0,    # No DACs
        'bits_per_weight': 8,       # Weights come from external memory (default precision)
        'input_precision_bits': 0,  # No input processing
        'tiles_per_chiplet': 0,     # No tiles
        'crossbars_per_tile': 0,    # No crossbars
        'threshold_temp': 330,      # Same thermal threshold as standard chiplet
    },
    # Add more chiplet types as needed
]
