import numpy as np
import os
import pandas as pd
import importlib
import os
from tensorflow.keras.models import model_from_json
import efficientnet.tfkeras
from tensorflow.keras.layers import Dense, Dropout, Flatten
from generator import AugmentedImageSequence
from tensorflow.keras.utils import OrderedEnqueuer

def get_enqueuer(csv,batch_size, FLAGS, tokenizer_wrapper, augmenter=None):
    data_generator = AugmentedImageSequence(
        dataset_csv_file=csv,
        class_names=FLAGS.csv_label_columns,
        tokenizer_wrapper=tokenizer_wrapper,
        source_image_dir=FLAGS.image_directory,
        batch_size=batch_size,
        target_size=FLAGS.image_target_size,
        augmenter=augmenter,
        shuffle_on_epoch_end=True,
    )
    enqueuer = OrderedEnqueuer(data_generator,
                               use_multiprocessing=False,
                               shuffle=False)
    return enqueuer, data_generator.steps


def get_layers(layer_sizes, activation = 'relu'):
    layers = []
    for layer_size in layer_sizes:
        if layer_size < 1:
            layers.append(Dropout(layer_size))
        else:
            layers.append(Dense(layer_size, activation=activation))
    return layers

def get_sample_counts(output_dir, dataset):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}"))
    total_count = df.shape[0]

    return total_count

def get_optimizer(optimizer_type, learning_rate, lr_decay=0):
    optimizer_class = getattr(importlib.import_module("tensorflow.keras.optimizers"), optimizer_type)
    optimizer = optimizer_class(lr=learning_rate, decay=lr_decay)
    return optimizer


def save_model(model, save_path, model_name):
    try:
        os.makedirs(save_path)
    except:
        print("path already exists")

    path = os.path.join(save_path, model_name)
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}.json".format(path), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}.h5".format(path))
    print("Saved model to disk")


def load_model(load_path, model_name):
    path = os.path.join(load_path, model_name)

    # load json and create model
    json_file = open('{}.json'.format(path), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # # load weights into new model
    loaded_model.load_weights("{}.h5".format(path))
    print("Loaded model from disk")
    return loaded_model