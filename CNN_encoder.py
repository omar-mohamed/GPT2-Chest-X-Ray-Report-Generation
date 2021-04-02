import tensorflow as tf
from utility import get_layers
from tensorflow.keras.models import Model
from utility import load_model
import numpy as np

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, model_path, model_name, pop_conv_layers, encoder_layers, tags_threshold, tags_embeddings=None,
                 finetune_visual_model=False):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        if tags_embeddings is not None:
            self.tags_embeddings=tf.Variable(shape=tags_embeddings.shape,initial_value=tags_embeddings, trainable=False, dtype=tf.float32)
        else:
            self.tags_embeddings=tf.Variable(shape=(105,400),initial_value=tf.ones((105,400)), trainable=False, dtype=tf.float32)
        self.encoder_layers = get_layers(encoder_layers, 'relu')
        visual_model = load_model(model_path, model_name)
        self.tags_threshold = tags_threshold
        self.visual_model = Model(inputs=visual_model.input,outputs=[visual_model.output, visual_model.layers[-pop_conv_layers-1].output], trainable=finetune_visual_model)

    def get_visual_features(self, images):

        predictions, visual_features = self.visual_model(images)
        predictions = tf.reshape(predictions, (predictions.shape[0], predictions.shape[-1], -1))
        visual_features = tf.reshape(visual_features, (visual_features.shape[0], -1, visual_features.shape[-1]))
        if self.tags_threshold >= 0:
            predictions =tf.cast(predictions >= self.tags_threshold, tf.float32)

        return predictions, visual_features

    def call(self, images):
        tags_predictions, visual_features = self.get_visual_features(images)
        if tags_predictions is not None:
            tags_embed = tf.multiply(tags_predictions,self.tags_embeddings)

        for layer in self.encoder_layers:
            visual_features = layer(visual_features)
            if tags_embed is not None:
              tags_embed = layer(tags_embed)
        return visual_features,tags_embed
