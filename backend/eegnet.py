from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Reshape
from tensorflow.keras.layers import Conv2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.constraints import max_norm

def EEGNet(nb_classes, Chans=14, Samples=1024, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    input1   = Input(shape=(Chans, Samples, 1))
    block1   = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1   = BatchNormalization()(block1)
    block1   = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1   = BatchNormalization()(block1)
    block1   = Activation('elu')(block1)
    block1   = AveragePooling2D((1, 4))(block1)
    block1   = Dropout(dropoutRate)(block1)
    block2   = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2   = BatchNormalization()(block2)
    block2   = Activation('elu')(block2)
    block2   = AveragePooling2D((1, 8))(block2)
    block2   = Dropout(dropoutRate)(block2)
    flat_shape = F2 * (Samples // 32)
    flatten  = Reshape((flat_shape,))(block2)
    dense    = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax  = Activation('softmax', name='softmax')(dense)
    return Model(inputs=input1, outputs=softmax)