import model.convert as convert
import numpy as np

Elements = {
    1: 'C', 2: 'N', 3: 'C', 4: 'O', 5: 'C',
    6: 'N', 7: 'C', 8: 'O', 9: 'N', 10: 'C',
    11: 'C', 12: 'C', 13: 'O', 14: 'O', 15: 'C',
    16: 'C', 17: 'C', 18: 'C', 19: 'C', 20: 'C',
    21: 'C', 22: 'H', 23: 'H', 24: 'H', 25: 'H',
    26: 'H', 27: 'H', 28: 'H', 29: 'H', 30: 'H',
    31: 'H', 32: 'H', 33: 'H', 34: 'H', 35: 'H',
    36: 'H', 37: 'H', 38: 'H', 39: 'H', 40: 'H'
}

Structure = {
    1: [1], 2: [2, 1], 3: [3, 2, 1], 4: [4, 3, 2, 1], 5: [5, 3, 2, 1],
    6: [6, 5, 3, 2], 7: [7, 1, 2, 3], 8: [8, 7, 1, 2], 9: [9, 7, 1, 2],
    10: [10, 9, 7, 1], 11: [11, 10, 9, 7], 12: [12, 10, 9, 7], 13: [13, 12, 10, 9],
    14: [14, 12, 10, 9], 15: [15, 1, 2, 3], 16: [16, 15, 1, 2], 17: [17, 16, 15, 1],
    18: [18, 17, 16, 15], 19: [19, 18, 17, 16], 20: [20, 19, 18, 17], 21: [21, 16, 15, 1],
    22: [22, 1, 2, 3], 23: [23, 2, 1, 3], 24: [24, 5, 3, 2], 25: [25, 5, 3, 2],
    26: [26, 6, 5, 3], 27: [27, 6, 5, 3], 28: [28, 9, 7, 1], 29: [29, 10, 9, 7],
    30: [30, 11, 10, 9], 31: [31, 11, 10, 9], 32: [32, 11, 10, 9], 33: [33, 13, 12, 10],
    34: [34, 15, 1, 2], 35: [35, 15, 1, 2], 36: [36, 17, 16, 15], 37: [37, 18, 17, 16],
    38: [38, 19, 18, 17], 39: [39, 20, 19, 18], 40: [40, 21, 20, 19]
}

Data = {
    'r2': 1.4426,
    'r3': 1.3547, 'a3': 122.33,
    'r4': 1.2325, 'a4': 124.21, 'd4': "{D11}",
    'r5': 1.5331, 'a5': 114.44, 'd5': "{d4} + 178.70",
    'r6': 1.4648, 'a6': 113.51, 'd6': "{D2}",
    'r7': 1.5294, 'a7': 108.35, 'd7': "{d15} + 120.75",
    'r8': 1.2329, 'a8': 121.63, 'd8': "{D4}",
    'r9': 1.3564, 'a9': 114.99, 'd9': "{d8} - 177.83",
    'r10': 1.45, 'a10': 121.45, 'd10': "{D12}",
    'r11': 1.5381, 'a11': 112.5, 'd11': "{D5}",
    'r12': 1.5191, 'a12': 107.5, 'd12': "{d11} + 122.45",
    'r13': 1.3559, 'a13': 111.52, 'd13': "{D7}",
    'r14': 1.2137, 'a14': 125.04, 'd14': "{d13} + 178.72",
    'r15': 1.5582, 'a15': 111.27, 'd15': "{D1}",
    'r16': 1.5041, 'a16': 112.82, 'd16': "{D9}",
    'r17': 1.4024, 'a17': 120.56, 'd17': "{D10}",
    'r18': 1.3953, 'a18': 120.74, 'd18': 180.0,
    'r19': 1.3962, 'a19': 120.17, 'd19': 0.0,
    'r20': 1.3959, 'a20': 119.61, 'd20': 0.0,
    'r21': 1.3957, 'a21': 120.09, 'd21': "{d17} + 180.00",
    'r22': 1.1008, 'a22': 108.24, 'd22': "{d15} + 241.31",
    'r23': 1.0212, 'a23': 119.28, 'd23': 180.0,
    'r24': 1.0998, 'a24': 105.95, 'd24': "{d6} + 240.51",
    'r25': 1.1024, 'a25': 106.36, 'd25': "{d6} + 127.59",
    'r26': 1.019, 'a26': 111.22, 'd26': "{D3}",
    'r27': 1.0176, 'a27': 111.41, 'd27': "{d26} + 119.53",
    'r28': 1.0185, 'a28': 120.46, 'd28': "{d10} - 162.51",
    'r29': 1.1021, 'a29': 108.93, 'd29': "{d11} + 239.85",
    'r30': 1.0955, 'a30': 108.42, 'd30': "{D6}",
    'r31': 1.0977, 'a31': 110.48, 'd31': "{d30} - 240.51",
    'r32': 1.0962, 'a32': 110.54, 'd32': "{d30} - 119.53",
    'r33': 0.977, 'a33': 106.28, 'd33': "{D8}",
    'r34': 1.099, 'a34': 106.2, 'd34': "{d16} + 121.79",
    'r35': 1.0991, 'a35': 107.67, 'd35': "{d16} - 122.35",
    'r36': 1.0921, 'a36': 119.33, 'd36': 0.0,
    'r37': 1.0901, 'a37': 119.8, 'd37': 180.0,
    'r38': 1.0899, 'a38': 120.15, 'd38': 180.0,
    'r39': 1.0906, 'a39': 120.15, 'd39': 180.0,
    'r40': 1.0918, 'a40': 120.03, 'd40': 180.0
}

Dimension = 9
Zero_Energy = -27467


def feature2atoms(f):
    dihe_angles = {
        "D1": f[0], "D2": f[1], "D3": f[2], "D4": f[3], "D5": f[4],
        "D7": f[5], "D8": f[6], "D9": f[7], "D10": f[8] / 2
    }
    dihe_angles["D6"] = -58.48 + 360
    dihe_angles["D11"] = 0.42
    dihe_angles["D12"] = 168.50

    data = convert.generate_data(Data, **dihe_angles)
    pos = convert.generate_position(Structure, data)
    return Elements, pos


def atoms2feature(e, pos):
    template = {"elements": Elements, "structure": Structure, "data": Data}
    e, pos = convert.auto_refine(
        e, pos, template, radius_error=0.15, angle_error=20)
    data = convert.generate_variable(Structure, pos)
    args = (15, 6, 26, 8, 11, 13, 33, 16, 17)
    feature = [data["d%d" % i] for i in args]
    feature[8] = (feature[8] * 2) % 360

    return feature
