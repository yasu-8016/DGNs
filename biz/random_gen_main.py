import datetime

import numpy as np
import pandas as pd
from rdkit import Chem

from biz.context_dc import Context
from biz.dgn_main import init_seed, log_dir_prep, logger_prep
from biz.mol_calc.smiles_prep import array2str2
from biz.molgan_module.metrics import MolecularMetrics as Mm


def batch_prep(ctx: Context):
    batch_ = np.random.normal(size=ctx.hypes.noise_shape)
    return batch_


def atom_num_prep(ctx: Context, ) -> int:
    min_num = ctx.hypes.atom_num_range[0]
    max_num = ctx.hypes.atom_num_range[1]
    atom_num = np.random.randint(low=min_num, high=max_num, size=1)

    return int(atom_num[0])


def do_gen(ctx: Context, df: pd.DataFrame, start_: str):

    min_num = ctx.hypes.atom_num_range[0]
    max_num = ctx.hypes.atom_num_range[1]

    ctx.paths.dir_name = f'Random_atoms{max_num}_20230226n'
    ctx.paths.model_name = f'Random_{start_}'
    ctx.paths.__post_init__()

    log_dir_prep(ctx=ctx)
    logger_prep(ctx=ctx)
    init_seed(ctx=ctx)

    counter = 0
    for atom_num in range(min_num, max_num+1, 1):

        for _ in range(10000):
            atoms = np.random.normal(size=(ctx.hypes.n_max, ctx.hypes.smiles_variation_num))

            smiles_ = array2str2(atoms, atom_num=atom_num, ctx=ctx)

            df.loc[counter, 'SMILES'] = smiles_
            df.loc[counter, 'atom_num'] = atom_num

            try:
                mol = Chem.MolFromSmiles(smiles_)

            except ValueError as err:
                print(f'{err}: Not able to parse {smiles_}')
                continue

            if mol:
                df.loc[counter, 'validity'] = 0

                df.loc[counter, 'solubility'] = Mm.water_octanol_partition_coefficient_scores([mol])[0]
                df.loc[counter, 'solubility_norm'] = Mm.water_octanol_partition_coefficient_scores([mol], norm=True)[0]
                df.loc[counter, 'qed'] = Mm.quantitative_estimation_druglikeness_scores([mol])[0]
                df.loc[counter, 'qed_norm'] = Mm.quantitative_estimation_druglikeness_scores([mol], norm=True)[0]
                df.loc[counter, 'sas'] = Mm.synthetic_accessibility_score_scores([mol])[0]
                df.loc[counter, 'sas_norm'] = Mm.synthetic_accessibility_score_scores([mol], norm=True)[0]
                df.loc[counter, 'sum_norm'] = sum([df.loc[counter, 'solubility_norm'],
                                              df.loc[counter, 'qed_norm'],
                                              df.loc[counter, 'sas_norm']])
            else:
                df.loc[counter, 'validity'] = 1

            counter += 1

    end_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(ctx.paths.csv / f'random_{start_}_{end_}.csv')


def main():
    ctx = Context()
    init_seed(ctx=ctx)
    col = ['SMILES', 'atom_num', 'validity',
           'solubility', 'solubility_norm', 'qed', 'qed_norm', 'sas', 'sas_norm', 'sum_norm']
    df = pd.DataFrame(columns=col)

    start_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    do_gen(ctx=ctx, df=df, start_=start_)


if __name__ == '__main__':
    main()
