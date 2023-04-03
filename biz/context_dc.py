import datetime
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from tensorflow.python.keras import Model


@dataclass
class HyperParams:
    # seeds
    seed: int = 0
    numpy_seed: int = 0
    tf_seed: int = 16
    op_seed: int = 16
    pd_seed: int = 0

    # operation
    iter_num: int = 2 ** 13  # Number of training epochs
    batch_iter: int = 2 ** 20
    batch_size: int = 10

    num_of_incorrect = 2
    num_of_correct = batch_size - num_of_incorrect
    commit_interval: int = 2

    fil_num: int = 64
    depth_g0: int = 2
    depth_g1: int = 4

    loss_weights_c = [1., 1., 1.]  # weights order: [solubility, qed, sas]
    loss_weights_g = [1., 1., 1.]  # weights order: [solubility, qed, sas]

    lr_g0cb0: float = 1e-6  # Learning rate
    lr_g0cb1: float = 1e-5  # Learning rate
    lr_c: float = 1e-4  # Learning rate
    lr_d: float = 1e-4  # Learning rate

    clip_g0cb0: float = 1e-6  # clip norm
    clip_g0cb1: float = 1e-5  # clip norm
    clip_c: float = 1e-4  # clip norm
    clip_d: float = 1e-4  # clip norm

    decay_g0cb0: float = 1e-3  # decay
    decay_g0cb1: float = 1e-3  # decay
    decay_c: float = 1e-3  # decay
    decay_d: float = 1e-3  # decay

    # molecule parameters
    n_max: int = 48  # 16の倍数のみ
    fixed_atom_num = None  # （水素を除く）原子の数を固定する場合に値を設定する。Noneとると以下のatom_num_rangeの範囲でランダムに生成される
    # atom_num_range: tuple = (3, 9)
    atom_num_range: tuple = (6, 20)
    # atom_num_range: tuple = (8, 40)

    init_atom_num: int = 1  # 生成機に初めに与えるランドダムに生成される分子の数

    atom_seed_list: list = field(default_factory=list)
    smiles_variation_num: int = field(default_factory=int)

    # Log P
    water_s_norm_target: float = 1.0
    water_s_target: float = 5.
    scale_log_p: float = 1.0

    # QED
    qed_norm_target: float = 0.80
    qed_target: float = 0.8
    scale_qed: float = 1.0

    # SA score
    sas_target: float = 2.
    sas_norm_target: float = 0.9
    scale_sas: float = 1.

    c_load_flag: bool = False
    g0_load_flag: bool = False
    g1_load_flag: bool = False

    target_label_num = 0

    # Cache
    max_cache_num: int = 1024
    cache_batch_size: int = 32

    def __post_init__(self):
        self.atom_seed_list = ['C', 'N', 'O',
                               'C=', 'N=', 'O=',
                               'C1', 'C2', 'C3',
                               'N1', 'N2', 'N3',
                               'C(', 'N(',
                               'C1(', 'C2(', 'C3(',
                               'N1(', 'N2(', 'N3(',
                               ]
        self.smiles_variation_num = len(self.atom_seed_list)
        self.frame_shape = (self.atom_num_range[1], self.smiles_variation_num)


@dataclass
class Batch:
    # grammar only
    mask_ng = None
    noise_ng = None
    pred_ng_smiles = None
    pred_ng_smiles_zeroone = None

    noise_total = None
    mask_total = None
    pred_total = None
    zeroone_total = None
    label = None
    target_label = None

    # both of grammar and gen_1
    mask = None
    noise = None
    pred_smiles = None
    pred_smiles_zeroone = None

    pred_xyz = None

    # for gen_1 and visualization
    atoms_from_smiles = None

    # calc only
    atom_num = None
    mols = {}
    smiles = {}

    # metrics
    water_s = None
    water_s_target = None

    sol_norm = None
    water_s_norm_target = None

    qed = None
    qed_target = None

    qed_norm = None
    qed_norm_target = None

    # sas
    sas = None
    sas_target = None

    sas_norm = None
    sas_norm_target = None

    sum_norm = None

    # flag
    correct_only: bool = False


@dataclass
class DataFrames:
    col_0 = ['SMILES', 'atom_num', 'validity',
             'water_s', 'water_s_norm',
             'qed', 'qed_norm',
             'sas', 'sas_norm',
             'sum_norm']
    col_1 = ['atoms_atom', 'atoms_xyz', 'atoms_adj', 'atoms_bond']
    col_2 = ['pred_atom', 'pred_xyz', 'pred_adj', 'pred_bond']
    col_3 = ['pred_smiles', 'pred_zeroone']
    col = col_0 + col_1 + col_2 + col_3

    best_: pd.DataFrame = field(default_factory=pd.DataFrame)
    cache_: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch_: pd.DataFrame = field(default_factory=pd.DataFrame)

    cache_len_max: int = 2 ** 10

    def __post_init__(self):
        self.best_ = pd.DataFrame(columns=self.col)
        self.cache_ = pd.DataFrame(columns=self.col)
        self.batch_ = pd.DataFrame(columns=self.col)


@dataclass
class Paths:
    training_data: Path = Path('')
    test_data: Path = Path('')

    saving_root: Path = field(default_factory=Path)
    saving_module: Path = field(default_factory=Path)
    saving_model: Path = field(default_factory=Path)

    saving_c_model: Path = field(default_factory=Path)
    saving_discriminator: Path = field(default_factory=Path)
    saving_g0_model: Path = field(default_factory=Path)
    saving_g1_model: Path = field(default_factory=Path)
    saving_comb0_model: Path = field(default_factory=Path)
    saving_comb1_model: Path = field(default_factory=Path)

    saving_log: Path = field(default_factory=Path)
    app_log: Path = field(default_factory=Path)

    tmp_im: Path = field(default_factory=Path)
    c_init_model: Path = field(default_factory=Path)
    g0_init_model: Path = field(default_factory=Path)
    g1_init_model: Path = field(default_factory=Path)

    csv: Path = field(default_factory=Path)
    model_name: str = 'DGNs'
    dir_name: str = 'DGNs'

    def __post_init__(self):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        name_ = now + self.model_name
        self.saving_root = Path('t_log') / self.dir_name / name_

        self.saving_module = self.saving_root / 'biz'
        self.saving_model = self.saving_root / 'model'
        self.saving_log = self.saving_root / 'log'
        self.app_log = self.saving_root / 'app_log'
        self.tmp_im = self.saving_root / 'tmp_im'
        self.init_models = self.saving_root / 'init_model'
        self.csv = self.saving_root / 'csv'

        self.saving_c_model = self.saving_model / f'c_model_{now}.h5'
        self.saving_g0_model = self.saving_model / f'g0_model_{now}.h5'
        self.saving_g1_model = self.saving_model / f'g1_model_{now}.h5'
        self.saving_comb0_model = self.saving_model / f'comb0_model_{now}.h5'
        self.saving_comb1_model = self.saving_model / f'comb1_model_{now}.h5'


@dataclass
class Models:
    initializer = None
    calc_model: Model = field(default_factory=Model)
    fixed_calc_model: Model = field(default_factory=Model)

    gen_0_model: Model = field(default_factory=Model)
    gen_1_model: Model = field(default_factory=Model)
    fixed_gen_1_model: Model = field(default_factory=Model)

    combined_0: Model = field(default_factory=Model)
    combined_1: Model = field(default_factory=Model)

    discriminator: Model = field(default_factory=Model)
    fixed_discriminator: Model = field(default_factory=Model)


@dataclass
class LossVal:
    loss_d: float = field(default_factory=float)
    loss_cb_0: float = field(default_factory=float)

    loss_c: list = field(default_factory=list)
    loss_g_1: list = field(default_factory=list)
    loss_cb_1: list = field(default_factory=list)

    loss_c_cache: list = field(default_factory=list)
    loss_g_1_cache: list = field(default_factory=list)

    calc_val: list = field(default_factory=list)


@dataclass
class Context:
    hypes: HyperParams = HyperParams()
    paths: Paths = Paths()

    models: Models = field(default_factory=Models)

    batch: Batch = field(default_factory=Batch)
    batch_prepared: bool = False

    loss_val: LossVal = field(default_factory=LossVal)

    logger = None

    df: DataFrames = field(default_factory=DataFrames)


def test():
    ctx = Context()
    print(ctx.paths.saving_log)
    print(ctx.paths.saving_model)
    print(ctx.paths.saving_module)
    print(ctx.paths.saving_comb0_model)


if __name__ == '__main__':
    test()
