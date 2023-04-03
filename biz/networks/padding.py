import numpy as np


def padding(inputs, shapes):
    # 0 padding
    n_in = []

    for _i0, _s0 in zip(inputs, shapes):  # inputsの種類で繰り返し

        dif_list = []
        for _i1, _s1 in zip(_i0.shape_a[1:], _s0):
            dif = _s1 - _i1
            dif_list.append(dif)

        if len(dif_list) == 3:
            padded = np.pad(_i0, ((0, 0), (0, dif_list[0]), (0, dif_list[1]), (0, dif_list[2])),
                            'constant', constant_values=0)
        elif len(dif_list) == 2:
            padded = np.pad(_i0, ((0, 0), (0, dif_list[0]), (0, dif_list[1]),), 'constant', constant_values=0)
        elif len(dif_list) == 1:
            padded = np.pad(_i0, ((0, 0), (0, dif_list[0]),), 'constant', constant_values=0)
        else:
            padded = _i0

        n_in.append(padded)

    return n_in
