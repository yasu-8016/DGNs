import datetime
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from rdkit import Chem

from biz.context_dc import Context
from biz.dgn_main import init_seed, log_dir_prep, logger_prep
from biz.mol_calc.smiles_cleansing import cleansing
from biz.molgan_module.metrics import MolecularMetrics as Mm
from biz.status import SampleMolStatusGA

"""
reference:
https://dse-souken.com/2021/05/25/ai-19/
some code was retrieved from above.
"""


@dataclass
class Parameters(Context):

    n_gen = 64  # number of generation
    pop_ = 64  # population
    cx_pb = 0.9  # crossover probability
    mut_pb = 0.1  # probability of mutation

    def __post_init__(self):
        self.atom_num_range = self.hypes.atom_num_range
        self.geneSet = self.hypes.atom_seed_list
        self.target_list = [self.hypes.water_s_norm_target,
                            self.hypes.qed_norm_target,
                            self.hypes.sas_norm_target]


def squared_error(guess, target):
    return (guess - target) ** 2


def get_fitness(guess, ):
    params = Parameters()

    _, calc_results = get_calc_results(guess=guess)
    target = params.target_list

    fitness = 0
    for r_, t_ in zip(calc_results, target, ):
        fitness += squared_error(r_, t_)
    return fitness,


def get_smiles(guess):
    params = Parameters()
    guess_list = []
    for n_ in guess:
        n_ = np.clip(n_, 0., len(params.geneSet) - 1)
        guess_list.append(params.geneSet[int(n_)])

    guess_str = ''.join(cleansing(guess_list))
    return guess_str


def get_calc_results(guess) -> tuple:
    guess_str = get_smiles(guess=guess)

    mol = None
    try:
        mol = Chem.MolFromSmiles(guess_str)

    except ValueError as err:
        print(f'{err}: Not able to parse {guess_str}')

    if mol:
        ws_norm = Mm.water_octanol_partition_coefficient_scores([mol], norm=True)[0]
        qed_norm = Mm.quantitative_estimation_druglikeness_scores([mol], norm=True)[0]
        sas_norm = Mm.synthetic_accessibility_score_scores([mol], norm=True)[0]

    else:
        ws_norm = 0.
        qed_norm = 0.
        sas_norm = 0.

    return mol, [ws_norm, qed_norm, sas_norm, ]


def do_ga(param: Parameters, atom_num):
    mu_ = [0.0 for _ in range(atom_num)]
    sigma_ = [20.0 for _ in range(atom_num)]

    creator.create('FitnessMin', base=base.Fitness, weights=(-1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMin)

    tb_ = base.Toolbox()
    tb_.register('attribute', random.randint, 0, len(param.geneSet) - 1)
    tb_.register('individual', tools.initRepeat, creator.Individual, tb_.attribute, n=atom_num)

    tb_.register("population", tools.initRepeat, list, tb_.individual)
    tb_.register("select", tools.selTournament, tournsize=5)
    tb_.register("mate", tools.cxBlend, alpha=0.2)
    tb_.register("mutate", tools.mutGaussian, mu=mu_, sigma=sigma_, indpb=0.2)
    tb_.register("evaluate", get_fitness)

    pop = tb_.population(n=param.pop_)
    for individual in pop:
        individual.fitness.values = tb_.evaluate(individual)

    hof = tools.ParetoFront()
    algorithms.eaSimple(pop, tb_,
                        cxpb=param.cx_pb, mutpb=param.mut_pb, ngen=param.n_gen, halloffame=hof,
                        verbose=True,
                        )
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind


def main():
    param = Parameters()
    atom_num_range = param.atom_num_range
    start_ = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ctx = Context()

    ctx.paths.dir_name = f'ga_atoms{atom_num_range[1]}_20230226n'
    ctx.paths.model_name = 'GA_'
    ctx.paths.__post_init__()

    col = ['SMILES', 'gen_num', 'atom_num', 'validity',
           'water_s', 'water_s_norm', 'qed', 'qed_norm', 'sas', 'sas_norm', 'sum_norm']

    df = pd.DataFrame(columns=col)

    counter = 0
    for atom_num_ in range(atom_num_range[0], atom_num_range[1]+1, 1):

        ctx.paths.model_name = f'GA_{atom_num_}'
        ctx.paths.__post_init__()

        log_dir_prep(ctx=ctx)
        logger_prep(ctx=ctx)
        init_seed(ctx=ctx)

        writer = tf.summary.create_file_writer(logdir=str(ctx.paths.saving_log))
        sample_writer = SampleMolStatusGA(writer=writer, ctx=ctx)

        for gen_num in range(10, 40, 1):

            param.n_gen = gen_num

            best_ind = do_ga(param=param, atom_num=atom_num_)

            smiles_ = get_smiles(best_ind)
            df.loc[counter, 'SMILES'] = smiles_
            df.loc[counter, 'atom_num'] = atom_num_

            mol, _ = get_calc_results(best_ind)

            if mol:
                df.loc[counter, 'validity'] = 0
                df.loc[counter, 'gen_num'] = gen_num

                sol = Mm.water_octanol_partition_coefficient_scores([mol])[0]
                sol_norm = Mm.water_octanol_partition_coefficient_scores([mol], norm=True)[0]
                qed = Mm.quantitative_estimation_druglikeness_scores([mol])[0]
                qed_norm = Mm.quantitative_estimation_druglikeness_scores([mol], norm=True)[0]
                sas = Mm.synthetic_accessibility_score_scores([mol])[0]
                sas_norm = Mm.synthetic_accessibility_score_scores([mol], norm=True)[0]
                sum_ = sum([sol_norm, qed_norm, sas_norm])
                df.loc[counter, 'solubility'] = sol
                df.loc[counter, 'solubility_norm'] = sol_norm
                df.loc[counter, 'qed'] = qed
                df.loc[counter, 'qed_norm'] = qed_norm
                df.loc[counter, 'sas'] = sas
                df.loc[counter, 'sas_norm'] = sas_norm
                df.loc[counter, 'sum_norm'] = sum_

                vals = [param.target_list[0], sol, sas_norm,
                        param.target_list[1], qed, qed_norm,
                        param.target_list[2], sas, sas_norm,
                        sum_]
                sample_writer.record_values(iter_num=gen_num, vals_=vals)
                sample_writer.record_images2(iter_num=gen_num, smiles=smiles_)

            else:
                df.loc[counter, 'validity'] = 1

            counter += 1

        end_ = datetime.datetime.now().strftime('%H%M%S')
        df.to_csv(ctx.paths.csv / f'ga_result_{start_}-{end_}.csv')


if __name__ == '__main__':
    main()
