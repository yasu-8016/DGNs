import datetime
import json
import os
import shutil
from logging import getLogger, config
from random import seed

import numpy as np
import pandas as pd
from rdkit import Chem

from biz.context_dc import Context
from biz.molgan_module.metrics import MolecularMetrics as Mm


def log_dir_prep(ctx: Context):
    os.makedirs(ctx.paths.saving_model)
    os.makedirs(ctx.paths.saving_log)
    os.makedirs(ctx.paths.app_log)
    os.makedirs(ctx.paths.tmp_im)
    os.makedirs(ctx.paths.csv)
    shutil.copytree(src='biz', dst=ctx.paths.saving_module)
    shutil.copytree(src='init_model', dst=ctx.paths.init_models)


def logger_prep(ctx: Context):
    with open('log_utils/log_config.json', 'r') as f:
        log_conf = json.load(f)

    log_conf["handlers"]["rotatingFileHandler"]["filename"] = str(ctx.paths.app_log / 'app.log')
    config.dictConfig(log_conf)
    ctx.logger = getLogger(__name__)


def init_seed(ctx: Context):
    seed(ctx.hypes.seed)
    np.random.seed(ctx.hypes.numpy_seed)

def count_atom_num(smiles_):
    atom_num = 0
    for c_ in smiles_:
        if c_ in 'CNOFcnof':
            atom_num += 1
        else:
            pass

    return atom_num


def do_gen(ctx: Context,
           df: pd.DataFrame,
           mols_: list,
           start_: str, verbose=True):
    ctx.paths.dir_name = f'qm9_statistics_20230308'
    ctx.paths.model_name = f'qm9_{start_}'
    ctx.paths.__post_init__()

    log_dir_prep(ctx=ctx)
    logger_prep(ctx=ctx)
    init_seed(ctx=ctx)

    counter = 0

    for mol_ in mols_:
        smiles_ = Chem.MolToSmiles(mol_)

        df.loc[counter, 'SMILES'] = smiles_
        df.loc[counter, 'atom_num'] = count_atom_num(smiles_)

        if mol_:
            df.loc[counter, 'validity'] = 0

            df.loc[counter, 'solubility'] = Mm.water_octanol_partition_coefficient_scores([mol_])[0]
            df.loc[counter, 'solubility_norm'] = Mm.water_octanol_partition_coefficient_scores([mol_], norm=True)[0]
            df.loc[counter, 'qed'] = Mm.quantitative_estimation_druglikeness_scores([mol_])[0]
            df.loc[counter, 'qed_norm'] = Mm.quantitative_estimation_druglikeness_scores([mol_], norm=True)[0]
            df.loc[counter, 'sas'] = Mm.synthetic_accessibility_score_scores([mol_])[0]
            df.loc[counter, 'sas_norm'] = Mm.synthetic_accessibility_score_scores([mol_], norm=True)[0]
            df.loc[counter, 'sum_norm'] = sum([df.loc[counter, 'solubility_norm'],
                                               df.loc[counter, 'qed_norm'],
                                               df.loc[counter, 'sas_norm']])
        else:
            df.loc[counter, 'validity'] = 1

        counter += 1
        if verbose:
            print(f'{counter}: {smiles_} processed')

    end_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(ctx.paths.csv / f'qm9_{start_}_{end_}.csv')


def op_():
    path_ = 'data/gdb9.sdf'
    suppl = Chem.SDMolSupplier(path_, removeHs=True)
    mols_ = [mol for mol in suppl if mol is not None]
    return mols_


def main():
    ctx = Context()
    init_seed(ctx=ctx)
    col = ['SMILES', 'atom_num', 'validity',
           'solubility', 'solubility_norm', 'qed', 'qed_norm', 'sas', 'sas_norm', 'sum_norm']
    df = pd.DataFrame(columns=col)

    start_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    mols_ = op_()
    do_gen(ctx=ctx, df=df, mols_=mols_, start_=start_)


def _atom_c_test():
    s_ = 'c1cccc1C(O)C'
    s_ = 'c1cnc1C(O)C'
    n_ = count_atom_num(s_)
    print(n_)


if __name__ == '__main__':
    main()
    # _atom_c_test()
