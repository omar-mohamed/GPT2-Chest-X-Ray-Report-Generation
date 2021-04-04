import tensorflow as tf
from CNN_encoder import CNN_Encoder
from configs import argHandler
import time
from medical_w2v_wrapper import Medical_W2V_Wrapper
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
from utility import get_optimizer, get_enqueuer
import os
import json
from augmenter import augmenter
from gpt2.gpt2_model import TFGPT2LMHeadModel
from test import evaluate_enqueuer
import pandas as pd

# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

FLAGS = argHandler()
FLAGS.setDefaults()
tf.keras.backend.set_learning_phase(1)

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                     FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

train_enqueuer, train_steps = get_enqueuer(FLAGS.train_csv, FLAGS.batch_size, FLAGS, tokenizer_wrapper)
test_enqueuer, test_steps = get_enqueuer(FLAGS.test_csv, 1, FLAGS, tokenizer_wrapper)
test_enqueuer2, test_steps2 = get_enqueuer(FLAGS.test_csv, FLAGS.batch_size, FLAGS, tokenizer_wrapper)

train_enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)

medical_w2v = Medical_W2V_Wrapper()
# medical_w2v.save_embeddings(tokenizer_wrapper.get_word_tokens_list(),FLAGS.tags)
# embeddings = medical_w2v.get_embeddings_matrix_for_words(tokenizer_wrapper.get_word_tokens_list(),
#                                                          FLAGS.tokenizer_vocab_size)
tags_embeddings = medical_w2v.get_embeddings_matrix_for_tags(FLAGS.tags)
# print(f"Embeddings shape: {embeddings.shape}")
print(f"Tags Embeddings shape: {tags_embeddings.shape}")

del medical_w2v

encoder = CNN_Encoder('pretrained_visual_models', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers, FLAGS.encoder_layers,
                      FLAGS.tags_threshold, tags_embeddings, FLAGS.finetune_visual_model)
decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)
optimizer = get_optimizer(FLAGS.optimizer_type, FLAGS.learning_rate)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, tokenizer_wrapper.GPT2_pad_token_id()))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


loss_plot = []


#
@tf.function
def train_step(images, target, test_mode=False):
    with tf.GradientTape() as tape:
        visual_features, tags_embeddings = encoder(images)
        dec_input = target[:, 0:-1]

        # passing the features through the decoder
        predictions, _ = decoder(dec_input, visual_features=visual_features, tags_embeddings=tags_embeddings, past=None)

        loss = loss_function(target[:, 1:], predictions)
    if not test_mode:
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss


ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

try:
    os.makedirs(FLAGS.ckpt_path)
except:
    print("path already exists")

with open(os.path.join(FLAGS.ckpt_path, 'configs.json'), 'w') as fp:
    json.dump(FLAGS, fp, indent=4)

ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

start_epoch = 0
best_test_avg_score = 0


def get_avg_score(scores_dict):
    avg_score = 0
    for value in scores_dict.values():
        avg_score += value
    avg_score = avg_score / len(scores_dict)
    return avg_score


if ckpt_manager.latest_checkpoint and FLAGS.continue_from_last_ckpt:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
    try:
        with open(os.path.join(FLAGS.ckpt_path, 'scores.json')) as scores_file:
            scores = json.load(scores_file)
            best_test_avg_score = get_avg_score(scores)
            print(f"best scores: {scores}")
    except:
        print("No previous scores found")

train_batch_losses_csv = {"step": [], "batch_loss": []}
test_batch_losses_csv = {"step": [], "batch_loss": []}
train_after_batch_losses_csv = {"step": [], "batch_loss": []}
losses_csv = {"epoch": [], "train_loss": [], "train_after_loss": [], "test_loss": []}
time_csv = {"epoch": [], 'time_taken': [], "scores": []}


def get_test_loss():
    global test_batch_losses_csv, losses_csv
    tf.keras.backend.set_learning_phase(0)

    test_enqueuer2.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
    test_generator = test_enqueuer2.get()

    batch_losses = []
    total_loss = 0
    step = 0
    for batch in range(test_steps2):
        img, target, _ = next(test_generator)
        batch_loss = train_step(img, target, True)
        test_batch_losses_csv['step'].append(step)
        test_batch_losses_csv['batch_loss'].append(batch_loss.numpy())
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        step += 1
    epoch_loss = total_loss / test_steps2
    test_enqueuer2.stop()
    tf.keras.backend.set_learning_phase(1)

    return epoch_loss, batch_losses


def get_train_loss(train_generator):
    global train_after_batch_losses_csv, losses_csv
    tf.keras.backend.set_learning_phase(0)

    batch_losses = []
    total_loss = 0
    step = 0
    for batch in range(train_steps):
        img, target, _ = next(train_generator)
        batch_loss = train_step(img, target, True)
        train_after_batch_losses_csv['step'].append(step)
        train_after_batch_losses_csv['batch_loss'].append(batch_loss.numpy())
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        step += 1
    epoch_loss = total_loss / train_steps
    tf.keras.backend.set_learning_phase(1)

    return epoch_loss, batch_losses


train_generator = train_enqueuer.get()

for epoch in range(start_epoch, FLAGS.num_epochs):
    start = time.time()
    total_loss = 0
    times_to_get_batch = 0
    pure_training_time = 0
    step = 0
    for batch in range(train_steps):
        t = time.time()
        img, target, _ = next(train_generator)
        # print("Time to get batch: {} s ".format(time.time() - t))
        if time.time() - t > 2:
            times_to_get_batch += 1
        step_time = time.time()
        batch_loss = train_step(img, target)
        pure_training_time += time.time() - step_time
        total_loss += batch_loss
        step += 1
        train_batch_losses_csv['step'].append(step)
        train_batch_losses_csv['batch_loss'].append(batch_loss.numpy())

        # print("Time to train step: {} s ".format(time.time() - t))

        if batch % 1 == 0 and batch > 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy()))
    # storing the epoch end loss value to plot later'
    total_loss = (total_loss / train_steps).numpy()
    loss_plot.append(total_loss)
    print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                        total_loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    print('Batches that took long: {}'.format(times_to_get_batch))
    test_epoch_loss, _ = get_test_loss()
    train_epoch_loss, _ = get_train_loss(train_generator)

    losses_csv["epoch"].append(epoch + 1)
    losses_csv['train_loss'].append(total_loss)
    losses_csv['train_after_loss'].append(train_epoch_loss.numpy())
    losses_csv['test_loss'].append(test_epoch_loss.numpy())
    pd.DataFrame(losses_csv).to_csv(os.path.join(FLAGS.ckpt_path, 'losses.csv'), index=False)
    pd.DataFrame(train_batch_losses_csv).to_csv(os.path.join(FLAGS.ckpt_path, 'train_batch_losses.csv'), index=False)
    pd.DataFrame(train_after_batch_losses_csv).to_csv(os.path.join(FLAGS.ckpt_path, 'train_after_batch_losses.csv'),
                                                      index=False)
    pd.DataFrame(test_batch_losses_csv).to_csv(os.path.join(FLAGS.ckpt_path, 'test_batch_losses.csv'), index=False)
    ckpt_manager.save()
    if epoch % FLAGS.epochs_to_evaluate == 0:
        current_avg_score = 0
        print("Evaluating on test set..")
        train_enqueuer.stop()
        current_scores = evaluate_enqueuer(test_enqueuer, test_steps, FLAGS, encoder, decoder, tokenizer_wrapper)
        time_csv['epoch'].append(epoch + 1)
        time_csv['time_taken'].append(pure_training_time)
        time_csv['scores'].append(current_scores)
        df = pd.DataFrame(time_csv)
        df.to_csv(os.path.join(FLAGS.ckpt_path, 'time.csv'), index=False)
        current_avg_score = get_avg_score(current_scores)
        train_enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
        if best_test_avg_score == 0 or current_avg_score > best_test_avg_score:
            ckpt_manager.save(fname='best_ckpt')
            print(f"found a new best model and saving the ckpt")
            best_test_avg_score = current_avg_score

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.savefig(FLAGS.ckpt_path + "/loss.png")

train_enqueuer.stop()
# plt.show()
