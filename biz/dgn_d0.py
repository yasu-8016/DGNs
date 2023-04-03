
import tensorflow as tf
from keras.layers import Input
from keras.models import Model, load_model
from keras.optimizers import Adam

from biz.batch_prep import GrammarBatch, Cache
from biz.context_dc import Context
from biz.dgn import DGNs
from biz.networks.dgn_models import discriminator, generator, calc_net_1
from biz.status import Cb0Status, CalcStatus, Cb1Status, DStatus, SampleMolStatus


class DGNandD0(DGNs):
    def __init__(self, ctx: Context):
        super().__init__(ctx=ctx)

    def build_models(self):
        ctx = self.ctx

        # optimizers
        opt_c = Adam(learning_rate=ctx.hypes.lr_c, clipnorm=ctx.hypes.clip_c, decay=ctx.hypes.decay_c)
        opt_d = Adam(learning_rate=ctx.hypes.lr_d, clipnorm=ctx.hypes.clip_d, decay=ctx.hypes.decay_d)
        opt_g0cb0 = Adam(learning_rate=ctx.hypes.lr_g0cb0, clipnorm=ctx.hypes.clip_g0cb0, decay=ctx.hypes.decay_g0cb0)
        opt_g0cb1 = Adam(learning_rate=ctx.hypes.lr_g0cb1, clipnorm=ctx.hypes.clip_g0cb1, decay=ctx.hypes.decay_g0cb1)

        # discriminator preparation
        # one-hot encoding smiles that is output of gen_0
        smiles_shape = (ctx.hypes.n_max, ctx.hypes.smiles_variation_num)

        d_in_0 = Input(shape=smiles_shape)
        d_in_1 = Input(shape=smiles_shape)
        c0_out = discriminator(x=[d_in_0, d_in_1], ctx=ctx)
        ctx.models.discriminator = Model(inputs=[d_in_0, d_in_1], outputs=c0_out, name='discriminator')
        ctx.models.fixed_discriminator = Model(inputs=[d_in_0, d_in_1], outputs=c0_out, name='fixed_discriminator')

        ctx.models.discriminator.compile(optimizer=opt_d, loss='binary_crossentropy')
        ctx.models.discriminator.summary()

        # gen_model_0 preparation
        g0_in_0a = Input(shape=smiles_shape)
        g0_in_1a = Input(shape=smiles_shape)
        g0_out_a = generator(x=[g0_in_0a, g0_in_1a], ctx=ctx)
        ctx.models.gen_0_model = Model(inputs=[g0_in_0a, g0_in_1a], outputs=g0_out_a, name='gen_0_model')
        ctx.models.gen_0_model.summary()

        # combined_0 model preparation
        g0_in_0b = Input(shape=smiles_shape)
        g0_in_1b = Input(shape=smiles_shape)
        g0_out_b = ctx.models.gen_0_model([g0_in_0b, g0_in_1b])
        ctx.models.fixed_discriminator.trainable = False

        pred_zeroone = Input(shape=smiles_shape)
        cb0_out = ctx.models.fixed_discriminator([g0_out_b, pred_zeroone])
        ctx.models.combined_0 = Model(inputs=[g0_in_0b, g0_in_1b, pred_zeroone],
                                      outputs=cb0_out, name='combined_0_model')
        ctx.models.combined_0.compile(optimizer=opt_g0cb0, loss='binary_crossentropy')
        ctx.models.combined_0.summary()

        # calc_model preparation
        g1_in_0 = Input(shape=smiles_shape)  # pred_smiles
        g1_in_1 = Input(shape=smiles_shape)  # pred_smiles_zeroone

        if ctx.hypes.g1_load_flag:
            ctx.models.calc_model = load_model(ctx.paths.c_init_model, custom_objects={'tf': tf})
            ctx.models.fixed_calc_model = load_model(ctx.paths.c_init_model, custom_objects={'tf': tf})
        else:
            g1_out_0, g1_out_1, g1_out_2 = calc_net_1([g1_in_0, g1_in_1, ], ctx=ctx)
            ctx.models.calc_model = Model(inputs=[g1_in_0, g1_in_1, ],
                                          outputs=[g1_out_0, g1_out_1, g1_out_2], name='calc_model')
            ctx.models.fixed_calc_model = Model(inputs=[g1_in_0, g1_in_1, ],
                                                outputs=[g1_out_0, g1_out_1, g1_out_2], name='fixed_calc_model')
        ctx.models.calc_model.compile(optimizer=opt_c, loss='logcosh', loss_weights=ctx.hypes.loss_weights_c)
        ctx.models.calc_model.summary()

        # combined model preparation
        g0_in_0c = Input(shape=smiles_shape)
        g0_in_1c = Input(shape=smiles_shape)
        g1_in_1b = Input(shape=smiles_shape)  # pred_smiles_zeroone

        # g0
        g0_out_c = ctx.models.gen_0_model([g0_in_0c, g0_in_1c])
        # g1
        ctx.models.fixed_calc_model.trainable = False
        c0_out_b = ctx.models.fixed_calc_model([g0_out_c, g1_in_1b, ])
        ctx.models.combined_1 = Model(inputs=[g0_in_0c, g0_in_1c, g1_in_1b, ],
                                      outputs=c0_out_b, name='combined_1_model')
        ctx.models.combined_1.compile(optimizer=opt_g0cb1, loss="logcosh", loss_weights=ctx.hypes.loss_weights_g)

        ctx.models.combined_1.summary()

    def train(self):
        ctx = self.ctx
        writer = tf.summary.create_file_writer(logdir=str(ctx.paths.saving_log))
        best_sum_loss_0 = 100.
        best_sum_loss_1 = 100.

        d_writer = DStatus(writer=writer, ctx=ctx)
        cb_0_writer = Cb0Status(writer=writer, ctx=ctx)
        c_writer = CalcStatus(writer=writer, ctx=ctx)
        cb_1_writer = Cb1Status(writer=writer, ctx=ctx)
        sample_writer = SampleMolStatus(writer=writer, ctx=ctx)

        cache_ = Cache(ctx)
        gb = GrammarBatch(ctx)

        for i_ in range(ctx.hypes.iter_num):
            gb.next_()
            if not ctx.batch.correct_only:
                ctx.loss_val.loss_d = ctx.models.discriminator.train_on_batch(x=[ctx.batch.pred_total,
                                                                                 ctx.batch.zeroone_total],
                                                                              y=ctx.batch.label)
                ctx.loss_val.loss_cb_0 = ctx.models.combined_0.train_on_batch(x=[ctx.batch.noise_total,
                                                                                 ctx.batch.mask_total,
                                                                                 ctx.batch.zeroone_total],
                                                                              y=ctx.batch.target_label)

                d_writer.record_values(iter_num=i_)
                cb_0_writer.record_values(iter_num=i_)
                ctx.logger.info(f'iteration #{i_}, d_loss: {ctx.loss_val.loss_d}, g_loss_0: {ctx.loss_val.loss_cb_0}, ')

            # cache
            cache_.append()

            ctx.loss_val.loss_c = ctx.models.calc_model.train_on_batch(x=[ctx.batch.pred_smiles,
                                                                          ctx.batch.pred_smiles_zeroone, ],
                                                                       y=[ctx.batch.sol_norm * ctx.hypes.scale_log_p,
                                                                          ctx.batch.qed_norm * ctx.hypes.scale_qed,
                                                                          ctx.batch.sas_norm * ctx.hypes.scale_sas,
                                                                          ])

            ctx.loss_val.loss_cb_1 = ctx.models.combined_1.train_on_batch(
                x=[ctx.batch.noise,
                   ctx.batch.mask,
                   ctx.batch.pred_smiles_zeroone,
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
                ctx.models.fixed_discriminator.set_weights(ctx.models.discriminator.get_weights())
                ctx.models.fixed_calc_model.set_weights(ctx.models.calc_model.get_weights())
                ctx.models.fixed_gen_1_model.set_weights(ctx.models.gen_1_model.get_weights())

            # saving_models
            condition_0 = i_ > 4 and best_sum_loss_0 > ctx.loss_val.loss_d + ctx.loss_val.loss_cb_0
            if condition_0:
                ctx.logger.info('Start Saving Models #0')
                ctx.models.discriminator.save(ctx.paths.saving_discriminator)
                ctx.models.gen_0_model.save(ctx.paths.saving_g0_model)
                ctx.models.combined_0.save(ctx.paths.saving_comb0_model)
                ctx.logger.info('Saving Finished #0')
                best_sum_loss_0 = ctx.loss_val.loss_d + ctx.loss_val.loss_cb_0

            if ctx.loss_val.loss_c and ctx.loss_val.loss_cb_1:
                condition_1 = i_ > 4 and best_sum_loss_1 > ctx.loss_val.loss_c[0] + ctx.loss_val.loss_cb_1[0]
                if condition_1:
                    ctx.logger.info('Start Saving Models #1')
                    ctx.models.calc_model.save(ctx.paths.saving_c_model)
                    ctx.models.gen_0_model.save(ctx.paths.saving_g0_model)
                    ctx.models.combined_1.save(ctx.paths.saving_comb0_model)
                    ctx.logger.info('Saving Finished #1')
                    best_sum_loss_1 = ctx.loss_val.loss_c[0] + ctx.loss_val.loss_cb_1[0]

            if i_ % 4:
                df_best = ctx.df.cache_.sort_values(by='sum_norm', ascending=False)
                df_best = df_best[ctx.df.col_0]
                if len(ctx.df.cache_) > 2 ** 10:
                    df_best.head(2 ** 10).to_csv(ctx.paths.csv / 'best.csv')
                else:
                    df_best.to_csv(ctx.paths.csv / 'best.csv')
