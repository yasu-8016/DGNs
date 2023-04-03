
from dataclasses import dataclass
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from biz.context_dc import Context

from biz.mol_calc.smiles_cleansing import cleansing


@dataclass
class Atom:
    H = [1, 0, 0, 0, 0]
    C = [0, 1, 0, 0, 0]
    N = [0, 0, 1, 0, 0]
    O = [0, 0, 0, 1, 0]
    F = [0, 0, 0, 0, 1]


def array2array(xyz_array, atom_num: int, ctx: Context):
    _atm_list = ctx.hypes.atom_seed_list
    zeroone_array = np.zeros_like(xyz_array)
    for _i1, _v1 in enumerate(xyz_array):
        _idx = np.argmax(_v1)
        zeroone_array[_i1][_idx] = 1
        if _i1 >= atom_num:
            break
    return zeroone_array


def array2str(xyz_array, atom_num: int, ctx: Context):
    _atm_list = ctx.hypes.atom_seed_list
    _s0 = ''
    for _i1, _v1 in enumerate(xyz_array):
        _idx = np.argmax(_v1)
        _s0 += _atm_list[_idx]
        if _i1 >= atom_num:
            break
    return _s0


def array2str2(xyz_array, atom_num: int, ctx: Context):
    _atm_list = ctx.hypes.atom_seed_list
    _s0 = []
    for _i1, _v1 in enumerate(xyz_array):
        _idx = np.argmax(_v1)
        _s0.append(_atm_list[_idx])
        if _i1 >= atom_num:
            break

    _s1 = cleansing(mol_list=_s0)

    return _s1


def mol2block(m: str):
    m1 = Chem.MolFromSmiles(m)
    xyz = Chem.AddHs(m1)

    AllChem.EmbedMolecule(xyz, randomSeed=0xf00d)
    mol_block = Chem.MolToMolBlock(xyz)
    return mol_block


def str2xyz(m: str, ctx: Context):
    """

    :param m: each molecule
    :param ctx:
    :return:
    """
    # print(f'Before Mol2Block: {m}')
    mol_block = mol2block(m)
    arr = mol_block.split(sep='\n')
    atms = arr[4:]

    atom = []
    atom_xyz = []
    bond = []
    for atm in atms:
        atm = atm.split(sep=' ')
        if len(atm) < 4:
            break
        elif 10 > len(atm) >= 4:
            tmp_list = [None, None, None]
            cnt = 0

            for _v in atm:
                if _v == '':
                    continue
                elif cnt < 3:
                    tmp_list[cnt] = int(_v)
                    cnt += 1
                else:
                    break
            bond.append(tmp_list)

        else:
            tmp_list2 = [None, None, None]
            cnt = 0

            for _v in atm:
                if _v == '':
                    continue

                elif cnt < 3:
                    tmp_list2[cnt] = float(_v)
                    cnt += 1

                elif cnt == 3:
                    a = None
                    if _v == 'c' or _v == 'C':
                        a = Atom.C
                    elif _v == 'O' or _v == 'o':
                        a = Atom.O
                    elif _v == 'N' or _v == 'n':
                        a = Atom.N
                    elif _v == 'F' or _v == 'f':
                        a = Atom.F
                    elif _v == 'H' or _v == 'h':
                        a = Atom.H
                    atom.append(a)
                    cnt += 1

                else:
                    break
            atom_xyz.append(tmp_list2)

    # zero padding
    pad_len = ctx.hypes.n_max - len(atom)
    if pad_len > 0:
        for _ in range(pad_len):
            atom.append([0, 0, 0, 0, 0])
            atom_xyz.append([0, 0, 0])
            bond.append([0, 0, 0])

    return np.array(atom), np.array(atom_xyz), np.array(bond)


def bond2adjacency(bond, ctx: Context) -> np.ndarray:
    adj = np.zeros((ctx.hypes.n_max, ctx.hypes.n_max))
    for _b in bond:
        adj[int(_b[0])][int(_b[1])] = 1

    return adj


def bond2edge(bond, ctx: Context) -> np.ndarray:
    edge = np.zeros((ctx.hypes.n_max, ctx.hypes.n_max, 4))
    for _b in bond:
        edge[int(_b[0]), int(_b[1]), int(_b[2])] = 1.

    return edge


def main():
    ctx = Context()
    m_ = np.eye(5, 5)
    print(array2str(m_, 4, ctx))


if __name__ == '__main__':
    main()
