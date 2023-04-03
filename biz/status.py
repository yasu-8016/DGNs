import numpy as np
import tensorflow as tf
from PIL import Image
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from biz.context_dc import Context


class Status:
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        self.writer = writer
        self.ctx = ctx
        self.m_name = []
        self.im_names = []

    def record_values(self, iter_num, ):
        pass


class DStatus(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(DStatus, self).__init__(writer, ctx)
        self.m_name = ['d_loss', ]

    def record_values(self, iter_num):
        vals_ = [self.ctx.loss_val.loss_d]
        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)


class Cb0Status(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(Cb0Status, self).__init__(writer, ctx)
        self.m_name = ['cb_0_loss', ]

    def record_values(self, iter_num):
        vals_ = [self.ctx.loss_val.loss_cb_0]
        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)


class G1Status(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(G1Status, self).__init__(writer, ctx)

        self.m_name = ['g1_loss_total', 'g1_loss_atom', 'g1_loss_xyz', 'g1_loss_adj', 'g1_loss_bond']

    def record_values(self, iter_num):
        vals_ = self.ctx.loss_val.loss_g_1
        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)


class CalcStatus(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(CalcStatus, self).__init__(writer, ctx)
        self.m_name = ['c_loss_total', 'c_loss_log_p', 'c_loss_qed', 'c_loss_sas', ]

    def record_values(self, iter_num):
        vals_ = self.ctx.loss_val.loss_c
        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)


class Cb1Status(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(Cb1Status, self).__init__(writer, ctx)
        self.m_name = ['cb1_loss_total', 'cb1_loss_log_p', 'cb1_loss_qed', 'cb1_loss_sas', ]

    def record_values(self, iter_num):
        vals_ = self.ctx.loss_val.loss_cb_1
        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)


class SampleMolStatus(Status):
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        super(SampleMolStatus, self).__init__(writer, ctx)
        self.m_name = ['target_solubility', 'score_solubility', 'score_solubility_norm',
                       'target_qed', 'score_qed', 'score_qed_norm',
                       'target_sas', 'score_sas', 'score_sas_norm',
                       'score_sum_norm',
                       ]
        self.im_names = ['smiles', ]

    def record_values(self, iter_num):

        vals_ = [
            self.ctx.batch.water_s_norm_target[0][0], self.ctx.batch.water_s[0][0], self.ctx.batch.sol_norm[0][0],
                 self.ctx.batch.qed_norm_target[0][0], self.ctx.batch.qed[0][0], self.ctx.batch.qed_norm[0][0],
                 self.ctx.batch.sas_norm_target[0][0], self.ctx.batch.sas[0][0],self.ctx.batch.sas_norm[0][0],
            self.ctx.batch.sum_norm[0][0],

                 ]

        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)

    def record_images(self, iter_num):
        im_names = ['smiles', ]
        image_list = [
            [self.ctx.batch.atoms_from_smiles[0], self.ctx.batch.atoms_from_smiles[1]],
        ]
        for _ims, _name in zip(image_list, im_names):

            if _ims:
                for batch_index, (atom, atom_xyz) in enumerate(zip(_ims[0], _ims[1])):
                    atom = atom.reshape((self.ctx.hypes.n_max, 5))
                    atom_xyz = atom_xyz.reshape((self.ctx.hypes.n_max, 3))

                    _sys = self.show_mol(atom, atom_xyz, _name)

                    _path_0 = self.ctx.paths.tmp_im / f'{_name}_im{batch_index}.png'
                    with open(_path_0, 'wb') as f:
                        write(f, _sys)

                    x_im = Image.open(_path_0)
                    _im = np.asarray(x_im)
                    _im = np.expand_dims(_im, axis=0)
                    with self.writer.as_default():
                        tf.summary.image(f'{_name}_im#{batch_index}', _im, step=iter_num)

                    if batch_index >= 2:
                        break

            else:
                break

    @staticmethod
    def show_mol(atom: np.ndarray, atom_xyz: np.ndarray, _name: str):
        sym_list = ['H', 'C', 'N', 'O', 'F']

        symbols = []
        for _d in atom:
            _index = np.argmax(_d)
            symbols.append(sym_list[_index])

        system = Atoms(positions=atom_xyz, symbols=symbols)
        return system

    def record_images2(self, iter_num):
        mols_ = []
        legends = []
        for k_, i_ in self.ctx.batch.smiles.items():
            mols_.append(Chem.MolFromSmiles(i_))
            legends.append(i_)

        if self.ctx.batch.mols:
            rdDepictor.SetPreferCoordGen(True)
            img = Draw.MolsToGridImage(mols_,
                                       molsPerRow=2,
                                       subImgSize=(200, 200),
                                       legends=legends
                                       )
            tmp_im_path = self.ctx.paths.tmp_im / 'im.png'
            img.save(tmp_im_path)

            x_im = Image.open(tmp_im_path)
            im_ = np.asarray(x_im)
            im_ = np.expand_dims(im_, axis=0)
            with self.writer.as_default():
                tf.summary.image('generated_molecules', im_, step=iter_num)


class SampleMolStatusGA:
    def __init__(self, writer: tf.summary.create_file_writer, ctx: Context, ):
        self.m_name = ['target_solubility', 'score_solubility', 'score_solubility_norm',
                       'target_qed', 'score_qed', 'score_qed_norm',
                       'target_sas', 'score_sas', 'score_sas_norm',
                       'score_sum_norm',
                       ]
        self.im_names = ['smiles', ]
        self.writer = writer
        self.ctx = ctx

    def record_values(self, iter_num, vals_):

        with self.writer.as_default():
            for name_, loss_ in zip(self.m_name, vals_):
                tf.summary.scalar(name=name_, data=loss_, step=iter_num)

    def record_images2(self, iter_num, smiles):
        rdDepictor.SetPreferCoordGen(True)
        mols_ = []
        legends = []
        for i_ in [smiles]:
            mols_.append(Chem.MolFromSmiles(i_))
            legends.append(i_)

        if mols_:
            img = Draw.MolsToGridImage(mols_,
                                       molsPerRow=2,
                                       subImgSize=(200, 200),
                                       legends=legends
                                       )
            tmp_im_path = self.ctx.paths.tmp_im / 'im.png'
            img.save(tmp_im_path)

            x_im = Image.open(tmp_im_path)
            im_ = np.asarray(x_im)
            im_ = np.expand_dims(im_, axis=0)
            with self.writer.as_default():
                tf.summary.image('generated_molecules', im_, step=iter_num)
