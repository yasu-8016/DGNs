
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam

from biz.batch_prep import BatchWithAtomZeros
from biz.context_dc import Context
from biz.dgn import DGNs
from biz.networks.dgn_models import generator, calc_net_mask_free
from biz.status import CalcStatus, Cb1Status, SampleMolStatus


class Ablation1(DGNs):
    """
    remove zero-mask from c-models
    """
    def __init__(self, ctx: Context):
        super().__init__(ctx=ctx)

    def build_models(self):
        ctx = self.ctx

        # optimizers
        opt_c = Adam(learning_rate=ctx.hypes.lr_c, clipnorm=ctx.hypes.clip_c, decay=ctx.hypes.decay_c)
        opt_g0cb1 = Adam(learning_rate=ctx.hypes.lr_g0cb1, clipnorm=ctx.hypes.clip_g0cb1, decay=ctx.hypes.decay_g0cb1)

        # discriminator preparation
        # one-hot encoding smiles that is output of gen_0
        smiles_shape = (ctx.hypes.n_max, ctx.hypes.smiles_variation_num)

        # gen_model_0 preparation
        g0_in_0a = Input(shape=smiles_shape)
        g0_in_1a = Input(shape=smiles_shape)
        g0_out_a = generator(x=[g0_in_0a, g0_in_1a], ctx=ctx)
        ctx.models.gen_0_model = Model(inputs=[g0_in_0a, g0_in_1a], outputs=g0_out_a, name='gen_0_model')
        ctx.models.gen_0_model.summary()

        # calc_model preparation
        g1_in_0 = Input(shape=smiles_shape)  # pred_smiles
        # g1_in_1 = Input(shape=smiles_shape)  # pred_smiles_zeroone

        if ctx.hypes.g1_load_flag:
            ctx.models.calc_model = load_model(ctx.paths.c_init_model, custom_objects={'tf': tf})
            ctx.models.fixed_calc_model = load_model(ctx.paths.c_init_model, custom_objects={'tf': tf})
        else:
            g1_out_0, g1_out_1, g1_out_2 = calc_net_mask_free(g1_in_0, ctx=ctx)
            ctx.models.calc_model = Model(inputs=g1_in_0,
                                          outputs=[g1_out_0, g1_out_1, g1_out_2], name='calc_model')
            ctx.models.fixed_calc_model = Model(inputs=g1_in_0,
                                                outputs=[g1_out_0, g1_out_1, g1_out_2], name='fixed_calc_model')
        ctx.models.calc_model.compile(optimizer=opt_c, loss='logcosh', loss_weights=ctx.hypes.loss_weights_c)
        ctx.models.calc_model.summary()

        # combined model preparation
        g0_in_0c = Input(shape=smiles_shape)
        g0_in_1c = Input(shape=smiles_shape)
        # g1_in_1b = Input(shape=smiles_shape)  # pred_smiles_zeroone

        # g0
        g0_out_c = ctx.models.gen_0_model([g0_in_0c, g0_in_1c])
        # g1
        ctx.models.fixed_calc_model.trainable = False
        c0_out_b = ctx.models.fixed_calc_model(g0_out_c)
        ctx.models.combined_1 = Model(inputs=[g0_in_0c, g0_in_1c, ],
                                      outputs=c0_out_b, name='combined_1_model')
        ctx.models.combined_1.compile(optimizer=opt_g0cb1, loss="logcosh", loss_weights=ctx.hypes.loss_weights_g)

        ctx.models.combined_1.summary()

    def train(self):
        ctx = self.ctx
        writer = tf.summary.create_file_writer(logdir=str(ctx.paths.saving_log))
        # best_sum_loss_0 = 100.
        best_sum_loss_1 = 100.

        # d_writer = DStatus(writer=writer, ctx=ctx)
        # cb_0_writer = Cb0Status(writer=writer, ctx=ctx)
        c_writer = CalcStatus(writer=writer, ctx=ctx)
        cb_1_writer = Cb1Status(writer=writer, ctx=ctx)
        sample_writer = SampleMolStatus(writer=writer, ctx=ctx)

        # cache_ = Cache(ctx)
        # gb = GrammarBatch(ctx)
        gb = BatchWithAtomZeros(ctx)

        for i_ in range(ctx.hypes.iter_num):
            gb.next_()

            ctx.loss_val.loss_c = ctx.models.calc_model.train_on_batch(x=[ctx.batch.pred_smiles,
                                                                          ],
                                                                       y=[ctx.batch.sol_norm * ctx.hypes.scale_log_p,
                                                                          ctx.batch.qed_norm * ctx.hypes.scale_qed,
                                                                          ctx.batch.sas_norm * ctx.hypes.scale_sas,
                                                                          ])

            ctx.loss_val.loss_cb_1 = ctx.models.combined_1.train_on_batch(
                x=[ctx.batch.noise,
                   ctx.batch.mask,
                   ],
                y=[ctx.batch.water_s_norm_target * ctx.hypes.scale_log_p,
                   ctx.batch.qed_norm_target * ctx.hypes.scale_qed,
                   ctx.batch.sas_norm_target * ctx.hypes.scale_sas,
                   ])

            ctx.logger.info(f'loss_c: {ctx.loss_val.loss_c}, loss_cb_1: {ctx.loss_val.loss_cb_1}, ')
            c_writer.record_values(iter_num=i_)
            cb_1_writer.record_values(iter_num=i_)
            sample_writer.record_values(iter_num=i_)
            sample_writer.record_images2(iter_num=i_)

            # copying weights from the trainable-model to the fixed-model
            if i_ % ctx.hypes.commit_interval == 0:
                ctx.models.fixed_calc_model.set_weights(ctx.models.calc_model.get_weights())

            if ctx.loss_val.loss_c and ctx.loss_val.loss_cb_1:
                condition_1 = i_ > 4 and best_sum_loss_1 > ctx.loss_val.loss_c[0] + ctx.loss_val.loss_cb_1[0]
                if condition_1:
                    ctx.logger.info('Start Saving Models #1')
                    ctx.models.calc_model.save(ctx.paths.saving_c_model)
                    ctx.models.gen_0_model.save(ctx.paths.saving_g0_model)
                    ctx.models.combined_1.save(ctx.paths.saving_comb0_model)
                    ctx.logger.info('Saving Finished #1')
                    best_sum_loss_1 = ctx.loss_val.loss_c[0] + ctx.loss_val.loss_cb_1[0]

