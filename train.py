import model
from datasets import Dataset
import yaml
import model_2 as model
import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # third gpu

if __name__ == '__main__':
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)

    epoch = cfg['training']['epoch']
    batch = cfg['training']['batch']

    dataset = Dataset('replayattack', batch_size=batch)
    GFA_CNN = model.get_model(dataset.numb_faceid)

    checkpoint_filepath = cfg['training']['checkpoint']
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    print('Preprocessing data ...')
    img_4_truth, label_truth, img_random_1, img_random_2, label_random, img_4_human_id, label_human_id = dataset.generate_batch()
    GFA_CNN.fit(
        img_4_human_id,
        label_human_id,
        epochs=25,
        batch_size=8,
        callbacks=[model_checkpoint_callback],
        # verbose=0,
    )

    ''' for mini batch dataset generating 
    
    for e in range(epoch):
        dataset.shuffle_dataset()
        
        for batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random, batch_img_4_human_id, batch_label_human_id in dataset.generate_minibatch():
            # print('batch_img_4_truth shape = ',batch_img_4_truth.shape)
            # print('batch_label_truth shape = ', batch_label_truth.shape)
            # print('batch_img_random shape = ', batch_img_random.shape)
            # print('batch_label_random shape = ', batch_label_random.shape)
            # print('batch_img_4_human_id shape = ', batch_img_4_human_id.shape)
            # print('batch_label_human_id shape = ', batch_label_human_id.shape)

            GFA_CNN.fit(
                {"as_input": batch_img_4_truth, "lpc_input_1": batch_img_random_1,"lpc_input_2": batch_img_random_2,
                 "facerecog_input": batch_img_4_human_id},
                {"as_output": batch_label_truth, 'lpc': batch_label_random, "fr_output": batch_label_human_id},
                epochs=1,
                batch_size=16,
                callbacks=[model_checkpoint_callback],
                # verbose=0,
            )
    '''
