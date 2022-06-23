import model.convert as convert
import numpy as np

Elements = {
    1: 'N', 2: 'C', 3: 'C', 4: 'O', 5: 'O', 6: 'C', 7: 'S',
    8: 'H', 9: 'H', 10: 'H', 11: 'H', 12: 'H', 13: 'H', 14: 'H'
}

Structure = {
    1: [1], 2: [2, 1], 3: [3, 2, 1], 4: [4, 3, 2, 1], 5: [5, 3, 2, 1],
    6: [6, 2, 1, 3], 7: [7, 6, 2, 1], 8: [8, 1, 2, 3], 9: [9, 1, 2, 3],
    10: [10, 2, 1, 3], 11: [11, 4, 3, 2], 12: [12, 6, 2, 1],
    13: [13, 6, 2, 1], 14: [14, 7, 6, 2]
}

Data = {
    'r2': 1.4559,
    'r3': 1.5159, 'a3': 108.34,
    'r4': 1.343, 'a4': 111.1, 'd4': "{D1}",
    'r5': 1.1981, 'a5': 126.31, 'd5': "{d4} + 179.36",
    'r6': 1.52, 'a6': 109.66, 'd6': 238.4,
    'r7': 1.8061, 'a7': 114.81, 'd7': "{D2}",
    'r8': 1.0105, 'a8': 110.78, 'd8': "{D3}",
    'r9': 1.0105, 'a9': 110.11, 'd9': "{d8} - 119.00",
    'r10': 1.1011, 'a10': 113.69, 'd10': 116.99,
    'r11': 0.9646, 'a11': 107.05, 'd11': "{D4}",
    'r12': 1.0912, 'a12': 108.49, 'd12': "{d7} - 116.89",
    'r13': 1.0884, 'a13': 110.31, 'd13': "{d7} + 125.47",
    'r14': 1.3405, 'a14': 96.22, 'd14': "{D5}"
}

Dimension = 5
Zero_Energy = -19635
# stable_energy = -19635.44


def feature2atoms(f):
    dihe_angles = {
        "D1": f[0],
        "D2": f[1],
        "D3": f[2],
        "D4": f[3],
        "D5": f[4]
    }
    data = convert.generate_data(Data, **dihe_angles)
    pos = convert.generate_position(Structure, data)
    return Elements, pos


def atoms2feature(e, pos):
    template = {"elements": Elements, "structure": Structure, "data": Data}
    e, pos = convert.auto_refine(
        e, pos, template, radius_error=0.15, angle_error=15)
    data = convert.generate_variable(Structure, pos)
    return [data["d%d" % i] for i in (4, 7, 8, 11, 14)]
