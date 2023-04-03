from keras.layers import Concatenate, BatchNormalization, Dense, Activation
from keras.layers import Flatten, Multiply, Conv1D
from biz.context_dc import Context
from biz.networks.block import SelfAttention, Unet, UnetWithSA


def generator(x, ctx: Context):
    ki_ = ctx.models.initializer
    random_noise, mask = x[0], x[1]

    x = Concatenate(axis=2)([random_noise, mask])
    x = BatchNormalization()(x)
    x = Conv1D(filters=ctx.hypes.fil_num, kernel_size=1,
               padding='same', strides=1, kernel_initializer=ki_)(x)

    x = Unet(ctx=ctx).call(x)

    x = SelfAttention(_filter=ctx.hypes.fil_num, ctx=ctx).call(x)

    x = Conv1D(filters=ctx.hypes.smiles_variation_num, kernel_size=1,
               padding='same', strides=1, kernel_initializer=ki_)(x)
    x = Activation('sigmoid')(x)
    x = Multiply()([x, mask])

    return x


def discriminator(x, ctx: Context):
    ki_ = ctx.models.initializer
    pred, zero_mask = x[0], x[1]
    x = Multiply()([pred, zero_mask])

    x = BatchNormalization()(x)
    x = Conv1D(filters=ctx.hypes.fil_num, kernel_size=1, padding='same', strides=1, kernel_initializer=ki_)(x)

    x = SelfAttention(_filter=ctx.hypes.fil_num, ctx=ctx).call(x)
    x = Unet(ctx=ctx).call(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(units=1, kernel_initializer=ki_)(x)
    x = Activation('sigmoid')(x)

    return x


def calc_net_1(_input, ctx: Context):
    ki_ = ctx.models.initializer
    x, n = _input[0], _input[1]
    x = Multiply()([x, n])
    x = Conv1D(filters=ctx.hypes.fil_num, kernel_size=1, padding='same', strides=1, kernel_initializer=ki_)(x)

    x = SelfAttention(_filter=ctx.hypes.fil_num, ctx=ctx).call(x)
    x = Unet(ctx=ctx).call(x)

    sol, qed, sas = x, x, x
    sol = Dense(units=1)(sol)  # log_p branch
    qed = Dense(units=1)(qed)  # qed branch
    sas = Dense(units=1)(sas)  # sas branch

    return [sol, qed, sas]


def discriminator_mask_free(x, ctx: Context):
    ki_ = ctx.models.initializer
    x = BatchNormalization()(x)
    x = Conv1D(filters=ctx.hypes.fil_num, kernel_size=1, padding='same', strides=1, kernel_initializer=ki_)(x)

    x = SelfAttention(_filter=ctx.hypes.fil_num, ctx=ctx).call(x)
    x = Unet(ctx=ctx).call(x)

    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(units=1, kernel_initializer=ki_)(x)
    x = Activation('sigmoid')(x)

    return x


def calc_net_mask_free(x, ctx: Context):
    ki_ = ctx.models.initializer
    x = Conv1D(filters=ctx.hypes.fil_num, kernel_size=1, padding='same', strides=1, kernel_initializer=ki_)(x)

    x = SelfAttention(_filter=ctx.hypes.fil_num, ctx=ctx).call(x)
    x = Unet(ctx=ctx).call(x)

    sol, qed, sas = x, x, x
    sol = Dense(units=1)(sol)  # log_p branch
    qed = Dense(units=1)(qed)  # qed branch
    sas = Dense(units=1)(sas)  # sas branch

    return [sol, qed, sas]
