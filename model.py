# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import math

def get_model(numb_faceid):
    # make share weight net first
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
    )
    last_layer = vgg16.get_layer('block5_pool')  # remove classify layer at top
    last_output = last_layer.output
    x = keras.layers.Flatten()(last_output)
    share_weight_net = keras.Model(vgg16.input, x)
    print(share_weight_net.summary())

    # input1 for anti spoofing
    as_input = keras.Input(shape=(224, 224, 3), name="as_input")
    as_flatten_1 = share_weight_net(as_input)
    as_fc = keras.Sequential(
        [
            keras.layers.Dense(4096, activation="relu", name="as_fc1"),
            keras.layers.Dense(4096, activation="relu", name="as_fc2"),
        ]
    )
    as_fc_i1 = as_fc(as_flatten_1)
    as_output = keras.layers.Dense(2, name="as_output", activation='softmax')(as_fc_i1)

    # input2 for anti spoofing
    lpc_input_1 = keras.Input(shape=(224, 224, 3), name="lpc_input_1")
    lpc_flatten_1 = share_weight_net(lpc_input_1)
    lpc_fc_o1 = as_fc(lpc_flatten_1)

    lpc_input_2 = keras.Input(shape=(224, 224, 3), name="lpc_input_2")
    lpc_flatten_2 = share_weight_net(lpc_input_2)
    lpc_fc_o2 = as_fc(lpc_flatten_2)

    lpc_layer = keras.layers.Lambda(lambda x: tf.math.square(x[0] - x[1]), name='lpc')
    lpc = lpc_layer([lpc_fc_o1, lpc_fc_o2])

    # second branch for face recognition
    fr_input = keras.Input(shape=(224, 224, 3), name="facerecog_input")
    fr_flatten = share_weight_net(fr_input)
    fr_fc = keras.Sequential(
        [
            keras.layers.Dense(4096, activation="relu", name="fr_fc1"),
            keras.layers.Dense(4096, activation="relu", name="fr_fc2"),
        ]
    )
    fr_fc_i = fr_fc(fr_flatten)
    fr_output = keras.layers.Dense(numb_faceid, name="fr_output", activation='softmax')(fr_fc_i)

    model = keras.Model(
        inputs=[as_input, lpc_input_1, lpc_input_2, fr_input],
        outputs=[as_output, lpc, fr_output],
    )
    # print(model.summary())

    model.compile(optimizer='adam',
                  loss={
                      "as_output": keras.losses.SparseCategoricalCrossentropy(),
                      "lpc": tpc_loss,
                      "fr_output": keras.losses.SparseCategoricalCrossentropy(),
                  },
                  loss_weights=[1, 2.5*math.exp(-5), 0.1],
                  metrics=['accuracy'])

    # dot_img_file = 'model_graph.png'
    # keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    return model


def tpc_loss(y_true, y_pred):  # y_true default is zero

    loss = tf.reduce_sum(y_pred, axis=1, keepdims=True)

    return loss


if __name__ == '__main__':

    model = get_model(20)
