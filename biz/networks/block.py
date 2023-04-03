
import tensorflow as tf
from keras.layers import Layer, Conv1D, BatchNormalization, Add, Activation, Conv1DTranspose

from biz.context_dc import Context


class Broadcast(Layer):
    def __init__(self, shape: tuple, ctx: Context):
        super(Broadcast, self).__init__()
        self.shape = (ctx.hypes.batch_size, )+shape

    def call(self, x, **kwargs):
        return tf.broadcast_to(x, self.shape)


class SelfAttention(Layer):
    def __init__(self, _filter: int, ctx: Context, data_format='channels_last', ):
        super(SelfAttention, self).__init__()
        self._filter = _filter
        self.data_format = data_format
        self.ki_ = ctx.models.initializer
        if self.data_format == 'channels_last':
            self.axis = -1
        else:
            self.axis = 1

    def call(self, x, **kwargs):
        f = Conv1D(filters=self._filter, kernel_size=1, strides=1, padding='same', data_format=self.data_format,
                   kernel_initializer=self.ki_)(x)
        g = Conv1D(filters=self._filter, kernel_size=1, strides=1, padding='same', data_format=self.data_format,
                   kernel_initializer=self.ki_)(x)
        h = Conv1D(filters=self._filter, kernel_size=1, strides=1, padding='same', data_format=self.data_format,
                   kernel_initializer=self.ki_)(x)

        s = tf.matmul(f, g, transpose_b=True)
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta, h)
        o = Conv1D(filters=self._filter, kernel_size=1, strides=1, padding='same', data_format=self.data_format,
                   kernel_initializer=self.ki_)(o)
        o = BatchNormalization(axis=self.axis)(o)

        x = Add()([o, x])

        return x


class ResidualBlock1D(Layer):
    def __init__(self, _filter: int, ctx: Context, kernel_=3, data_format='channels_last'):
        super(ResidualBlock1D, self).__init__()
        self._filter = _filter
        self.ctx = ctx
        self.ki_ = ctx.models.initializer

        self.kernel_ = kernel_
        self.data_format = data_format

        if self.data_format == 'channels_last':
            self.axis = -1
        else:
            self.axis = 1

    def call(self, x, **kwargs):
        res = x

        x = BatchNormalization(axis=self.axis)(x)
        x = Conv1D(filters=self._filter, kernel_size=self.kernel_, strides=1, padding='same',
                   data_format=self.data_format, kernel_initializer=self.ki_)(x)
        x = Activation('tanh')(x)

        x = BatchNormalization(axis=self.axis)(x)
        x = Conv1D(filters=self._filter, kernel_size=self.kernel_, strides=1, padding='same',
                   data_format=self.data_format, kernel_initializer=self.ki_)(x)
        x = Activation('tanh')(x)

        x = Add()([x, res])

        return x


class Unet(Layer):
    def __init__(self, ctx: Context):
        super(Unet, self).__init__()
        self.ctx = ctx

    def call(self, x, **kwargs):

        ctx = self.ctx
        ki_ = ctx.models.initializer

        # 1/1
        res_1_1 = x
        f_num_1_1 = ctx.hypes.fil_num

        # 1/2
        f_num_1_2 = ctx.hypes.fil_num * 2
        x = Conv1D(filters=f_num_1_2, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_2, ctx=ctx).call(x)
        res_1_2 = x

        # 1/2 * 1/2 = 1/4
        f_num_1_4 = ctx.hypes.fil_num * 4
        x = Conv1D(filters=f_num_1_4, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_4, ctx=ctx).call(x)
        res_1_4 = x

        # 1/2 * 1/2 * 1/2 = 1/8
        f_num_1_8 = ctx.hypes.fil_num * 8
        x = Conv1D(filters=f_num_1_8, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_8, ctx=ctx).call(x)

        # res-branches
        for _ in range(ctx.hypes.depth_g1):
            res_1_1 = ResidualBlock1D(_filter=f_num_1_1, ctx=ctx).call(res_1_1)
            res_1_2 = ResidualBlock1D(_filter=f_num_1_2, ctx=ctx).call(res_1_2)
            res_1_4 = ResidualBlock1D(_filter=f_num_1_4, ctx=ctx).call(res_1_4)

        # 1/4
        x = Conv1DTranspose(filters=f_num_1_4, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_4])
        x = BatchNormalization()(x)

        # 1/2
        x = Conv1DTranspose(filters=f_num_1_2, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_2])
        x = BatchNormalization()(x)

        # 1/1
        x = Conv1DTranspose(filters=f_num_1_1, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_1])

        return x



class UnetWithSA(Layer):
    def __init__(self, ctx: Context):
        super(UnetWithSA, self).__init__()
        self.ctx = ctx

    def call(self, x, **kwargs):

        ctx = self.ctx
        ki_ = ctx.models.initializer

        # 1/1
        res_1_1 = x
        f_num_1_1 = ctx.hypes.fil_num

        # Self attention
        x = SelfAttention(_filter=f_num_1_1, ctx=ctx).call(x)

        # 1/2
        f_num_1_2 = ctx.hypes.fil_num * 2
        x = Conv1D(filters=f_num_1_2, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_2, ctx=ctx).call(x)
        res_1_2 = x

        # 1/2 * 1/2 = 1/4
        f_num_1_4 = ctx.hypes.fil_num * 4
        x = Conv1D(filters=f_num_1_4, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_4, ctx=ctx).call(x)
        res_1_4 = x

        # 1/2 * 1/2 * 1/2 = 1/8
        f_num_1_8 = ctx.hypes.fil_num * 8
        x = Conv1D(filters=f_num_1_8, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        for _ in range(ctx.hypes.depth_g1):
            x = ResidualBlock1D(_filter=f_num_1_8, ctx=ctx).call(x)

        # res-branches
        for _ in range(ctx.hypes.depth_g1):
            res_1_1 = ResidualBlock1D(_filter=f_num_1_1, ctx=ctx).call(res_1_1)
            res_1_2 = ResidualBlock1D(_filter=f_num_1_2, ctx=ctx).call(res_1_2)
            res_1_4 = ResidualBlock1D(_filter=f_num_1_4, ctx=ctx).call(res_1_4)

        # 1/4
        x = Conv1DTranspose(filters=f_num_1_4, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_4])
        x = BatchNormalization()(x)

        # 1/2
        x = Conv1DTranspose(filters=f_num_1_2, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_2])
        x = BatchNormalization()(x)

        # 1/1
        x = Conv1DTranspose(filters=f_num_1_1, kernel_size=3, padding='same', strides=2, kernel_initializer=ki_)(x)
        x = Activation('tanh')(x)
        x = Add()([x, res_1_1])

        return x
