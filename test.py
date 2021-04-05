import tensorflow as tf
from CNN_encoder import CNN_Encoder
import os
from configs import argHandler
from tokenizer_wrapper import TokenizerWrapper
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from caption_evaluation import get_evalutation_scores
from utility import get_enqueuer
import numpy as np
from PIL import Image
import json
import time
from gpt2.gpt2_model import TFGPT2LMHeadModel
import pandas as pd
from tqdm import tqdm

# input_ids=None,
# visual_features = None,
# tags_embedding = None,
# max_length=None,
# min_length=None,
# do_sample=True,
# early_stopping=False,
# num_beams=None,
# temperature=None,
# top_k=None,
# top_p=None,
# repetition_penalty=None,
# bos_token_id=None,
# pad_token_id=None,
# eos_token_ids=None,
# length_penalty=None,
# no_repeat_ngram_size=None,
# num_return_sequences=None,
# attention_mask=None,
avg_time = 0
step_n = 1


def evaluate_full(FLAGS, encoder, decoder, tokenizer_wrapper, images):
    global avg_time, step_n
    visual_features, tags_embeddings = encoder(images)
    dec_input = tf.expand_dims(tokenizer_wrapper.GPT2_encode("startseq", pad=False), 0)
    # dec_input = tf.tile(dec_input,[images.shape[0],1])
    num_beams = FLAGS.beam_width

    visual_features = tf.tile(visual_features, [num_beams, 1, 1])
    tags_embeddings = tf.tile(tags_embeddings, [num_beams, 1, 1])
    start_time = time.time()
    tokens = decoder.generate(dec_input, max_length=FLAGS.max_sequence_length, num_beams=num_beams, min_length=3,
                              eos_token_ids=tokenizer_wrapper.GPT2_eos_token_id(), no_repeat_ngram_size=0,
                              visual_features=visual_features,
                              tags_embedding=tags_embeddings, do_sample=False, early_stopping=True)
    end_time = time.time() - start_time
    # print(f"Step time: {end_time}")
    avg_time += end_time
    # print(f"avg Step time: {avg_time / step_n}")

    sentence = tokenizer_wrapper.GPT2_decode(tokens[0])
    sentence = tokenizer_wrapper.filter_special_words(sentence)
    step_n += 1
    return sentence


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def save_output_prediction(FLAGS, img_name, target_sentence, predicted_sentence):
    if not os.path.exists(FLAGS.output_images_folder):
        os.makedirs(FLAGS.output_images_folder)

    image_path = os.path.join(FLAGS.image_directory, img_name)

    img = mpimg.imread(os.path.join(image_path))

    caption = "Real caption: {}\n\nPrediction: {}".format(target_sentence, predicted_sentence)
    # plt.ioff()
    fig = plt.figure(figsize=(7.20, 10.80))
    fig.add_axes((.0, .5, .9, .7))
    fig.text(.1, .3, caption, wrap=True, fontsize=20)

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.savefig(FLAGS.output_images_folder + "/{}".format(img_name))
    plt.close(fig)


def evaluate_enqueuer(enqueuer, FLAGS, encoder, decoder, tokenizer_wrapper, name='Test set', verbose=True,
                      write_json=True, write_images=False, test_mode=False):
    tf.keras.backend.set_learning_phase(0)
    hypothesis = []
    references = []
    if not enqueuer.is_running():
        enqueuer.start(workers=FLAGS.generator_workers, max_queue_size=FLAGS.generator_queue_length)
    start = time.time()
    csv_dict = {"image_path": [], "real": [], "prediction": []}
    generator = enqueuer.get()
    for batch in tqdm(list(range(generator.steps))):
        images, target, img_path = next(generator)

        predicted_sentence = evaluate_full(FLAGS, encoder, decoder, tokenizer_wrapper,
                                           images)

        csv_dict["prediction"].append(predicted_sentence)
        csv_dict["image_path"].append(os.path.basename(img_path[0]))
        target_sentence = tokenizer_wrapper.GPT2_decode(target[0])
        target_sentence = tokenizer_wrapper.filter_special_words(target_sentence)
        csv_dict["real"].append(target_sentence)
        target_word_list = tokenizer_wrapper.GPT2_format_output(target_sentence)
        references.append([target_word_list])
        hypothesis_word_list = tokenizer_wrapper.GPT2_format_output(predicted_sentence)
        if hypothesis_word_list[-1] == hypothesis_word_list[-1]:
            hypothesis_word_list = hypothesis_word_list[:-1]
        hypothesis.append(hypothesis_word_list)
        if write_images:
            save_output_prediction(FLAGS, img_path[0], target_sentence, predicted_sentence)
        # print('Time taken for saving image {} sec\n'.format(time.time() - t))

    enqueuer.stop()
    scores = get_evalutation_scores(hypothesis, references, test_mode)
    print("{} scores: {}".format(name, scores))
    if write_json:
        with open(os.path.join(FLAGS.ckpt_path, 'scores.json'), 'w') as fp:
            json.dump(str(scores), fp, indent=4)
    print('Time taken for evaluation {} sec\n'.format(time.time() - start))
    tf.keras.backend.set_learning_phase(1)
    df = pd.DataFrame(csv_dict)
    df.to_csv(FLAGS.ckpt_path + "/predictions.csv", index=False)
    return scores


if __name__ == "__main__":
    FLAGS = argHandler()
    FLAGS.setDefaults()

    tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                         FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

    print("** load test generator **")

    test_enqueuer, test_steps = get_enqueuer(FLAGS.test_csv, 1, FLAGS, tokenizer_wrapper)
    test_enqueuer.start(workers=1, max_queue_size=8)

    encoder = CNN_Encoder('pretrained_visual_model', FLAGS.visual_model_name, FLAGS.visual_model_pop_layers,
                          FLAGS.encoder_layers, FLAGS.tags_threshold)

    decoder = TFGPT2LMHeadModel.from_pretrained('distilgpt2', from_pt=True, resume_download=True)

    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.ckpt_path, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Restored from checkpoint: {}".format(ckpt_manager.latest_checkpoint))
    evaluate_enqueuer(test_enqueuer, FLAGS, encoder, decoder, tokenizer_wrapper, write_images=True, test_mode=True)
