import json
import os
import shutil
from logging import getLogger, config
from random import seed

import numpy as np
import tensorflow as tf

from biz.dgn import DGNs
from biz.ablation_1 import Ablation1
from biz.context_dc import Context, Models
from biz.dgn_d0 import DGNandD0


def log_dir_prep(ctx: Context):
    os.makedirs(ctx.paths.saving_model)
    os.makedirs(ctx.paths.saving_log)
    os.makedirs(ctx.paths.app_log)
    os.makedirs(ctx.paths.tmp_im)
    os.makedirs(ctx.paths.csv)
    shutil.copytree(src='biz', dst=ctx.paths.saving_module)
    shutil.copytree(src='init_model', dst=ctx.paths.init_models)


def target_prep(ctx: Context):
    b_num_c = ctx.hypes.num_of_correct
    ctx.batch.water_s_norm_target = np.ones((b_num_c, 1)) * ctx.hypes.water_s_norm_target
    ctx.batch.qed_norm_target = np.ones((b_num_c, 1)) * ctx.hypes.qed_norm_target
    ctx.batch.sas_norm_target = np.ones((b_num_c, 1)) * ctx.hypes.sas_norm_target

    ctx.batch.water_s_target = np.ones((b_num_c, 1)) * ctx.hypes.water_s_target
    ctx.batch.qed_target = np.ones((b_num_c, 1)) * ctx.hypes.qed_target
    ctx.batch.sas_target = np.ones((b_num_c, 1)) * ctx.hypes.sas_target


def logger_prep(ctx: Context):
    with open('log_utils/log_config.json', 'r') as f:
        log_conf = json.load(f)

    log_conf["handlers"]["rotatingFileHandler"]["filename"] = str(ctx.paths.app_log / 'app.log')
    config.dictConfig(log_conf)
    ctx.logger = getLogger(__name__)


def init_seed(ctx: Context):
    seed(ctx.hypes.seed)
    np.random.seed(ctx.hypes.numpy_seed)
    tf.random.set_seed(ctx.hypes.tf_seed)

    operation_lebel_seed = ctx.hypes.op_seed
    ctx.models.initializer = tf.keras.initializers.GlorotUniform(seed=operation_lebel_seed)


def main():
    model_dict = {
        # 'DGNs_without_D0': DGNs,
        'DGNs_wo_ZM_D0': Ablation1,
        # 'DGN_with_D0nZM': DGNandD0,
    }
    tf_seeds_list = range(0, 4, 1)
    # tf_seeds_list = [22, 21, 17, 16]
    for i_ in tf_seeds_list:
        for k_, m_ in model_dict.items():

            ctx = Context()
            ctx.hypes.tf_seed = i_
            ctx.paths.model_name = k_+'_'+str(i_)
            ctx.paths.dir_name = f'a-num{ctx.hypes.atom_num_range[1]}_20230228_iniAtom{ctx.hypes.init_atom_num}'
            ctx.paths.__post_init__()

            log_dir_prep(ctx=ctx)
            logger_prep(ctx=ctx)
            ctx.logger.info(f'######### {k_}, tf_seed: {i_} started ##############')

            init_seed(ctx=ctx)
            ctx.models = Models()

            dgn_model = m_(ctx=ctx)
            dgn_model.build_models()
            target_prep(ctx=ctx)
            dgn_model.train()


if __name__ == '__main__':
    main()
