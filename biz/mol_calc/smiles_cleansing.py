from dataclasses import dataclass

import numpy as np


@dataclass
class Atoms:
    atom_seed_list = ['C', 'N', 'O',
                      'c',
                      'c1', 'C1',
                      'c2', 'C2',
                      'c3', 'C3',
                      'C(', 'N(',
                      ]


def even_flag(mol_list: list, char_: str) -> bool:
    mol_str = "".join(mol_list)
    par_num = mol_str.count(char_)
    return True if par_num % 2 == 0 else False


def gen_mol(atom_num: int = 8):
    ctx = Atoms()
    a_list = ctx.atom_seed_list
    z = np.random.randint(0, len(a_list), atom_num)

    mol_list = []
    for a_ in z:
        mol_list.append(a_list[a_])
    return mol_list


def cleansing(mol_list: list):
    par_num_even = even_flag(mol_list, '(')
    par_counter = 0

    num_list = ['1', '2', '3']
    for i_, s_ in enumerate(mol_list):
        if i_+2 <= len(mol_list) - 1:
            for num_ in num_list:
                if num_ in s_ and num_ in mol_list[i_+2]:
                    mol_list[i_+2] = mol_list[i_+2].replace(num_, '')
        if i_ + 1 <= len(mol_list) - 1:
            for num_ in num_list:
                if num_ in s_ and num_ in mol_list[i_+1]:
                    mol_list[i_+1] = mol_list[i_+1].replace(num_, '')


    num_even_dict = {}
    for num_ in ['1', '2', '3']:
        num_even_dict[num_] = even_flag(mol_list, num_)
    counter_dict = {'1': 0, '2': 0, '3': 0}

    mol_n_list = []
    par_list = []
    for i_, s_ in enumerate(mol_list):

        if '(' in s_:
            if not par_num_even and par_counter == 0:
                s_ = s_.replace('(', '')
            elif not par_list:
                par_list.append(')')
            else:
                s_ = s_.replace('(', ')')
                par_list.remove(')')

            par_counter += 1

        for k_, v_ in num_even_dict.items():
            if k_ in s_:
                if not v_ and counter_dict[k_] == 0:
                    s_ = s_.replace(k_, '')
                counter_dict[k_] += 1

        mol_n_list.append(s_)

    return ''.join(mol_n_list)


def main():
    for i_ in range(16):
        np.random.seed(seed=i_)
        mol_list = gen_mol(atom_num=10)
        print(f'mol smiles: {mol_list}')
        print(f'modified mol: {cleansing(mol_list)}')


if __name__ == '__main__':
    main()
