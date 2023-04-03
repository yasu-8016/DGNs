import random

import numpy as np
from rdkit import Chem

from biz.context_dc import Context
from biz.mol_calc.smiles_prep import str2xyz, bond2adjacency, bond2edge, array2array, array2str2
from biz.molgan_module.metrics import MolecularMetrics as Mm


class GrammarBatch:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        n_max = self.ctx.hypes.n_max
        b_num_c = self.ctx.hypes.num_of_correct
        b_num_ng = self.ctx.hypes.num_of_incorrect

        # shape for grammar batch
        pred_smiles_shape = (b_num_c, n_max, ctx.hypes.smiles_variation_num)
        pred_ng_smiles_shape = (b_num_ng, n_max, ctx.hypes.smiles_variation_num)
        pred_smiles_shape_total = (b_num_c + b_num_ng, n_max, ctx.hypes.smiles_variation_num)

        # shapes for calc batch
        atom_array_shape = (b_num_c, n_max, 5)
        xyz_array_shape = (b_num_c, n_max, 3)
        adj_array_shape = (b_num_c, n_max, n_max)
        edge_array_shape = (b_num_c, n_max, n_max, 4)

        # initialize
        # grammar only
        self.ctx.batch.mask_ng = np.empty(pred_ng_smiles_shape)
        self.ctx.batch.noise_ng = np.empty(pred_ng_smiles_shape)
        self.ctx.batch.pred_ng_smiles = np.zeros(pred_ng_smiles_shape)
        self.ctx.batch.pred_ng_smiles_zeroone = np.zeros(pred_ng_smiles_shape)

        self.ctx.batch.noise_total = np.empty(pred_smiles_shape_total)
        self.ctx.batch.mask_total = np.empty(pred_smiles_shape_total)
        self.ctx.batch.pred_total = np.empty(pred_smiles_shape_total)
        self.ctx.batch.zeroone_total = np.empty(pred_smiles_shape_total)
        self.ctx.batch.label = np.zeros((b_num_c + b_num_ng, 1))
        self.ctx.batch.target_label = np.ones((b_num_c + b_num_ng, 1)) * self.ctx.hypes.target_label_num

        # both of grammar and calc
        self.ctx.batch.mask = np.empty(pred_smiles_shape)
        self.ctx.batch.noise = np.empty(pred_smiles_shape)
        self.ctx.batch.pred_smiles = np.zeros(pred_smiles_shape)
        self.ctx.batch.pred_smiles_zeroone = np.zeros(pred_smiles_shape)

        # calc only
        self.ctx.batch.atom_num = np.zeros((b_num_c, 1))
        self.ctx.batch.smiles = {}
        self.ctx.batch.mols = {}

        self.ctx.batch.water_s = np.zeros((b_num_c, 1))
        self.ctx.batch.sol_norm = np.zeros((b_num_c, 1))
        self.ctx.batch.qed = np.zeros((b_num_c, 1))
        self.ctx.batch.qed_norm = np.zeros((b_num_c, 1))
        self.ctx.batch.sas = np.zeros((b_num_c, 1))
        self.ctx.batch.sas_norm = np.zeros((b_num_c, 1))
        self.ctx.batch.sum_norm = np.zeros((b_num_c, 1))

        # for visualization
        self.ctx.batch.atoms_from_smiles = [np.zeros(atom_array_shape),
                                            np.zeros(xyz_array_shape),
                                            np.zeros(adj_array_shape),
                                            np.zeros(edge_array_shape)]

    def next_(self):

        self.ctx.batch.num_of_correct = 0

        c_num = self.ctx.hypes.num_of_correct
        i_num = self.ctx.hypes.num_of_incorrect

        correct_counter = 0
        incorrect_counter = 0

        for _iter in range(self.ctx.hypes.batch_iter):

            if correct_counter < c_num:

                noise = self.noise_prep()
                atom_num = self.atom_num_prep()
                mask_ = self.mask_prep(atom_num_=atom_num)
                pred_smiles_ = self.pred_smiles_prep(noise_=noise, mask_=mask_)
                smiles_ = self.get_smiles(pred_smiles_[0], atom_num_=atom_num)
                pred_smiles_zeroone = self.get_smiles_zeroone(pred_smiles_=pred_smiles_[0], atom_num_=atom_num)
                try:
                    # batch クラスに格納する前に、文法の適否判断をする
                    mol_ = self.get_mol(smiles_)
                    if not mol_:
                        raise ValueError
                    water_s = self.solubility_calc(mol_, norm=False)
                    water_s_norm = self.solubility_calc(mol_, norm=True)
                    qed_ = self.qed_calc(mol_, norm=False)
                    qed_norm = self.qed_calc(mol_, norm=True)
                    sas_ = self.sas_calc(mol_, norm=False)
                    sas_norm = self.sas_calc(mol_, norm=True)
                    sum_norm = water_s_norm + qed_norm + sas_norm

                    if correct_counter < c_num:
                        self.ctx.batch.noise[correct_counter] = noise
                        self.ctx.batch.atom_num[correct_counter] = atom_num
                        self.ctx.batch.mask[correct_counter] = mask_

                        self.ctx.batch.pred_smiles[correct_counter] = pred_smiles_[0]
                        self.ctx.batch.pred_smiles_zeroone[correct_counter] = pred_smiles_zeroone
                        self.ctx.batch.smiles[correct_counter] = smiles_
                        self.ctx.batch.mols[correct_counter] = mol_

                        self.ctx.batch.water_s[correct_counter] = water_s
                        self.ctx.batch.sol_norm[correct_counter] = water_s_norm
                        self.ctx.batch.qed[correct_counter] = qed_
                        self.ctx.batch.qed_norm[correct_counter] = qed_norm
                        self.ctx.batch.sas[correct_counter] = sas_
                        self.ctx.batch.sas_norm[correct_counter] = sas_norm

                        self.ctx.batch.sum_norm[correct_counter] = sum_norm

                        # for gen_1 and visualization
                        atoms_ = self.atoms_prep_from_smiles(smiles_)
                        self.ctx.batch.atoms_from_smiles[0][correct_counter] = atoms_[0]
                        self.ctx.batch.atoms_from_smiles[1][correct_counter] = atoms_[1]
                        self.ctx.batch.atoms_from_smiles[2][correct_counter] = atoms_[2]
                        self.ctx.batch.atoms_from_smiles[3][correct_counter] = atoms_[3]

                        correct_counter += 1
                        self.ctx.logger.info(f'valid grammar. counter c:{correct_counter}, i:{incorrect_counter}')
                    else:
                        continue

                except Exception as err:
                    self.ctx.logger.error(f'{err}: SMILES Grammar is incorrect; #{_iter}', exc_info=True)
                    if incorrect_counter < i_num:
                        self.ctx.batch.noise_ng[incorrect_counter] = noise
                        self.ctx.batch.mask_ng[incorrect_counter] = mask_
                        self.ctx.batch.pred_ng_smiles[incorrect_counter] = pred_smiles_
                        self.ctx.batch.pred_ng_smiles_zeroone[incorrect_counter] = pred_smiles_zeroone
                        incorrect_counter += 1
                        self.ctx.logger.info(f'invalid grammar. counter c:{correct_counter}, i:{incorrect_counter}')
                    else:
                        continue

            else:
                break
        if incorrect_counter < i_num:
            self.ctx.batch.correct_only = True
            self.ctx.logger.info(f'correct-only batch prepared')
        else:
            self.total_prep()
            self.ctx.batch.correct_only = False
            self.ctx.logger.info(f'total-batch prepared')

    def total_prep(self):
        c_num = self.ctx.hypes.num_of_correct
        i_num = self.ctx.hypes.num_of_incorrect
        total_index = list(range(c_num + i_num))
        random.shuffle(total_index)
        for i0, i_ in enumerate(total_index):
            if i_ < c_num:
                self.ctx.batch.noise_total[i0] = self.ctx.batch.noise[i_]
                self.ctx.batch.mask_total[i0] = self.ctx.batch.mask[i_]
                self.ctx.batch.pred_total[i0] = self.ctx.batch.pred_smiles[i_]
                self.ctx.batch.zeroone_total[i0] = self.ctx.batch.pred_smiles_zeroone[i_]
                self.ctx.batch.label[i0] = 0

            else:
                self.ctx.batch.noise_total[i0] = self.ctx.batch.noise_ng[i_ - c_num]
                self.ctx.batch.mask_total[i0] = self.ctx.batch.mask_ng[i_ - c_num]
                self.ctx.batch.pred_total[i0] = self.ctx.batch.pred_ng_smiles[i_ - c_num]
                self.ctx.batch.zeroone_total[i0] = self.ctx.batch.pred_ng_smiles_zeroone[i_ - c_num]
                self.ctx.batch.label[i0] = 1


    def noise_prep(self):
        return np.random.normal(size=(self.ctx.hypes.n_max, self.ctx.hypes.smiles_variation_num))

    def atom_num_prep(self) -> int:
        if self.ctx.hypes.fixed_atom_num:
            atom_num_ = self.ctx.hypes.fixed_atom_num
        else:
            atom_num_ = np.random.randint(low=self.ctx.hypes.atom_num_range[0],
                                          high=self.ctx.hypes.atom_num_range[1], size=1)
        return int(atom_num_[0])

    def mask_prep(self, atom_num_, ):
        mask_one = np.ones((atom_num_, self.ctx.hypes.smiles_variation_num))
        mask_zero = np.zeros((self.ctx.hypes.n_max - atom_num_, self.ctx.hypes.smiles_variation_num))
        return np.concatenate([mask_one, mask_zero], axis=0)

    def pred_smiles_prep(self, noise_, mask_):
        input_0 = np.expand_dims(noise_, axis=0)
        input_1 = np.expand_dims(mask_, axis=0)
        return self.ctx.models.gen_0_model.predict_on_batch([input_0, input_1])

    def get_smiles(self, pred_smiles_, atom_num_) -> str:
        return array2str2(pred_smiles_, int(atom_num_), ctx=self.ctx)

    def get_smiles_zeroone(self, pred_smiles_, atom_num_):
        return array2array(pred_smiles_, int(atom_num_), ctx=self.ctx)

    @staticmethod
    def get_mol(smiles_: str) -> Chem.Mol:
        return Chem.MolFromSmiles(smiles_)

    def atoms_prep_from_smiles(self, smiles_: str) -> list:
        atom, atom_xyz, bond = str2xyz(smiles_, ctx=self.ctx)
        adj_array = bond2adjacency(bond, ctx=self.ctx)
        edge_array = bond2edge(bond, ctx=self.ctx)
        return [atom, atom_xyz, adj_array, edge_array]

    @staticmethod
    def solubility_calc(mol, norm):
        return Mm.water_octanol_partition_coefficient_scores([mol], norm=norm)[0]

    @staticmethod
    def qed_calc(mol, norm):
        return Mm.quantitative_estimation_druglikeness_scores([mol], norm=norm)[0]

    @staticmethod
    def sas_calc(mol, norm):
        return Mm.synthetic_accessibility_score_scores([mol], norm=norm)[0]


class Cache:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    def append(self):
        dfc = self.ctx.df.cache_
        len_ = len(dfc)
        for i_ in range(self.ctx.hypes.num_of_correct):
            dfc.loc[len_+i_, 'SMILES'] = self.ctx.batch.smiles[i_]
            dfc.loc[len_+i_, 'atom_num'] = self.ctx.batch.atom_num[i_][0]

            dfc.loc[len_+i_, 'water_s'] = self.ctx.batch.water_s[i_][0]
            dfc.loc[len_+i_, 'water_s_norm'] = self.ctx.batch.sol_norm[i_][0]
            dfc.loc[len_+i_, 'qed'] = self.ctx.batch.qed[i_][0]
            dfc.loc[len_+i_, 'qed_norm'] = self.ctx.batch.qed_norm[i_][0]
            dfc.loc[len_+i_, 'sas'] = self.ctx.batch.sas[i_][0]
            dfc.loc[len_+i_, 'sas_norm'] = self.ctx.batch.sas_norm[i_][0]
            dfc.loc[len_+i_, 'sum_norm'] = self.ctx.batch.sum_norm[i_][0]

            dfc.loc[len_+i_, 'atoms_atom'] = self.ctx.batch.atoms_from_smiles[0][i_]
            dfc.loc[len_+i_, 'atoms_xyz'] = self.ctx.batch.atoms_from_smiles[1][i_]
            dfc.loc[len_+i_, 'atoms_adj'] = self.ctx.batch.atoms_from_smiles[2][i_]
            dfc.loc[len_+i_, 'atoms_bond'] = self.ctx.batch.atoms_from_smiles[3][i_]

            dfc.loc[len_+i_, 'pred_smiles'] = self.ctx.batch.pred_smiles[i_]
            dfc.loc[len_+i_, 'pred_zeroone'] = self.ctx.batch.pred_smiles_zeroone[i_]

        dfc.drop_duplicates(subset='SMILES', inplace=True)  # 重複排除
        self.ctx.df.cache_ = dfc.sort_values(by='sum_norm', ascending=False)
        self.ctx.df.cache_.reset_index(drop=True, inplace=True)

        if len(dfc) > self.ctx.hypes.max_cache_num:
            self.ctx.df.cache_ = self.ctx.df.cache_.head(self.ctx.hypes.max_cache_num)

    def next_batch(self):
        if len(self.ctx.df.cache_) >= self.ctx.hypes.cache_batch_size:
            self.ctx.df.batch_ = self.ctx.df.cache_.sample(n=self.ctx.hypes.cache_batch_size, )
        else:
            self.ctx.df.batch_ = self.ctx.df.cache_


class BatchWithInitAtom(GrammarBatch):
    def __init__(self, ctx: Context):
        super().__init__(ctx=ctx)
        
    def mask_prep(self, atom_num_, ):
        init_atoms = self.prep_init_atoms()
        mask_one = np.ones((atom_num_ - self.ctx.hypes.init_atom_num, self.ctx.hypes.smiles_variation_num))
        mask_zero = np.zeros((self.ctx.hypes.n_max - atom_num_, self.ctx.hypes.smiles_variation_num))
        mask = np.concatenate([init_atoms, mask_one, mask_zero], axis=0)
        return mask

    def prep_init_atoms(self):
        init_atom_num = self.ctx.hypes.init_atom_num
        init_atoms = np.zeros((init_atom_num, self.ctx.hypes.smiles_variation_num))
        atom_index = np.random.randint(low=0, high=self.ctx.hypes.smiles_variation_num, size=init_atom_num)
        for i_, a_ in enumerate(atom_index):
            init_atoms[i_, a_] = 1
        return init_atoms


class BatchWithAtomZeros(BatchWithInitAtom):
    def __init__(self, ctx: Context):
        super().__init__(ctx=ctx)

    def noise_prep(self):
        noise_ = np.zeros(shape=(self.ctx.hypes.n_max - self.ctx.hypes.init_atom_num,
                                 self.ctx.hypes.smiles_variation_num))
        init_atom_ = self.prep_init_atoms()

        return np.concatenate([init_atom_, noise_], axis=0)


def main_test():
    ctx = Context()

    nb = BatchWithInitAtom(ctx)
    num = nb.atom_num_prep()
    mask = nb.mask_prep(num)
    noise = nb.noise_prep()

    mul = noise * mask
    print(mul)
    smiles = nb.get_smiles(mul, num)
    print(smiles)

    zeroone = nb.get_smiles_zeroone(mul, num)
    print(zeroone)


if __name__ == '__main__':
    main_test()
