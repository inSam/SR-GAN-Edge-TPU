from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, Conv2DTranspose, DepthwiseConv2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
import tensorflow as tf

from model.common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11

LR_SIZE = 24
HR_SIZE = 96


def upsample(x_in, dt_size, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)

    #B,H,W,N = tf.shape(x)
    #scale = 2
    x = tf.reshape(tf.transpose(tf.reshape(x, [-1, dt_size, dt_size, 2, num_filters//2]), perm=[0,1,3,2,4]), [-1, dt_size*2,dt_size*2, num_filters//4])
    #x = tf.reshape(x, [-1, dt_size*2,dt_size*2, num_filters//4])
    #print(x)
    #x = Lambda(pixel_shuffle(scale=2))(x)
    #x = Conv2DTranspose(num_filters//4, kernel_size=1, strides=(2,2), padding='same')(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(lambda x: x)(x_in)
    #x = Lambda(normalize_01)(x_in)
    
    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, HR_SIZE, num_filters * 4)
    x = upsample(x, HR_SIZE*2, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    
    #x = Lambda(denormalize_m11)(x)
    return Model(x_in, x)


generator = sr_resnet


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64):
    x_in = Input(shape=(HR_SIZE, HR_SIZE, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)
