import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
import numpy as np
from transformers import GPT2Tokenizer


class TokenizerWrapper:
    def __init__(self, dataset_csv_file, class_name, max_caption_length, tokenizer_num_words=None):
        dataset_df = pd.read_csv(dataset_csv_file)
        sentences = dataset_df[class_name].tolist()
        self.max_caption_length = max_caption_length
        self.tokenizer_num_words = tokenizer_num_words
        self.init_tokenizer(sentences)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
        self.gpt2_tokenizer.pad_token = "<"

    def clean_sentence(self, sentence):
        return text_to_word_sequence(sentence, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    def GPT2_pad_token_id(self):
        return self.gpt2_tokenizer.pad_token_id
    def GPT2_eos_token_id(self):
        return self.gpt2_tokenizer.eos_token_id

    def GPT2_encode(self, sentences, pad=True, max_length=None):
        if max_length is None:
            max_length = self.max_caption_length
        if isinstance(sentences, str):
            return self.gpt2_tokenizer.encode(sentences, add_special_tokens=True, max_length=max_length,
                                              pad_to_max_length=pad)
        tokens = np.zeros((sentences.shape[0], max_length), dtype=int)

        for i in range(len(sentences)):
            if pd.isna(sentences[i]):
                sentences[i][0] = ""
            sentence = sentences[i][0].lower()
            sentence = sentence.replace('"', '')
            sentence = sentence.replace('xxxx', '')
            sentence = sentence.replace('endseq','<|endoftext|>')
            tokens[i] = self.gpt2_tokenizer.encode(sentence, add_special_tokens=True,
                                                   max_length=max_length, pad_to_max_length=pad)
        return tokens

    def GPT2_decode(self, tokens):
        return self.gpt2_tokenizer.decode(tokens, skip_special_tokens=True)

    def GPT2_format_output(self, sentence):
        sentence = self.clean_sentence(sentence)
        return sentence

    def filter_special_words(self, sentence):
        sentence = sentence.replace('startseq', '')
        sentence = sentence.replace('endseq', '')
        sentence = sentence.replace('<|endoftext|>', '')
        sentence = sentence.replace('<', '')
        sentence = sentence.strip()
        return sentence

    def init_tokenizer(self, sentences):

        for i in range(len(sentences)):
            if pd.isna(sentences[i]):
                sentences[i] = ""
            sentences[i] = self.clean_sentence(sentences[i])

        # Tokenize the reviews
        print("Tokenizing dataset..")
        self.tokenizer = Tokenizer(oov_token='UNK', num_words=self.tokenizer_num_words)
        self.tokenizer.fit_on_texts(sentences)  # give each word a unique id
        print("number of tokens: {}".format(self.tokenizer.word_index))
        print("Tokenizing is complete.")

    def get_tokenizer_num_words(self):
        return self.tokenizer_num_words

    def get_token_of_word(self, word):
        return self.tokenizer.word_index[word]

    def get_word_from_token(self, token):
        try:
            return self.tokenizer.index_word[token]
        except:
            return ""

    def get_sentence_from_tokens(self, tokens):
        sentence = []
        for token in tokens[0]:
            word = self.get_word_from_token(token)
            if word == 'endseq':
                return sentence
            if word != 'startseq':
                sentence.append(word)

        return sentence

    def get_string_from_word_list(self, word_list):

        return " ".join(word_list)

    def get_word_tokens_list(self):
        return self.tokenizer.word_index

    def tokenize_sentences(self, sentences):
        index = 0
        tokenized_sentences = np.zeros((sentences.shape[0], self.max_caption_length), dtype=int)
        for caption in sentences:
            tokenized_caption = self.tokenizer.texts_to_sequences([self.clean_sentence(caption[0])])
            tokenized_sentences[index] = pad_sequences(tokenized_caption, maxlen=self.max_caption_length,
                                                       padding='post')  # padded with max length
            index = index + 1
        return tokenized_sentences
