import pandas as pd
from configs import argHandler
from caption_evaluation import get_evalutation_scores
from tokenizer_wrapper import TokenizerWrapper
import re
import collections
from copy import deepcopy
FLAGS = argHandler()
FLAGS.setDefaults()

df = pd.read_csv('predictions.csv')
labels =  df['real']
preds =  df['prediction']

tokenizer_wrapper = TokenizerWrapper(FLAGS.all_data_csv, FLAGS.csv_label_columns[0],
                                     FLAGS.max_sequence_length, FLAGS.tokenizer_vocab_size)

def tokenize(string):
    """Convert string to lowercase and split into words (ignoring
    punctuation), returning list of words.
    """
    return re.findall(r'\w+', string.lower())

def count_ngrams(lines, min_length=3, max_length=3):
    """Iterate through given lines iterator (file object or list of
    lines) and return n-gram frequencies. The return value is a dict
    mapping the length of the n-gram to a collections.Counter
    object of n-gram tuple and number of times that n-gram occurred.
    Returned dict includes n-grams of length min_length to max_length.
    """
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)

    # Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1

    # Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                    add_queue()

    # Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()

    return ngrams


def print_most_frequent(ngrams, num=10):
    """Print num most common n-grams of each length in n-grams dict."""
    for n in sorted(ngrams):
        print('----- {} most common {}-grams -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')

def filter_words(word_list,filters):
    filter_len = len(filters)
    filtered = []
    i = 0
    counter = 0
    while(i<len(word_list)):
        if tuple(word_list[i:i+filter_len]) == filters:
            if counter >=1:
                i+=filter_len
            else:
                filtered.extend(word_list[i:i+filter_len])
                counter = 1
                i +=filter_len
        else:
            filtered.append(word_list[i])
            i += 1
    return filtered


def remove_ngrams(word_list,ngrams,n_filter=2):
    filtered = deepcopy(word_list)
    for n in sorted(ngrams,reverse=True):
        for gram, count in ngrams[n].most_common(10):
            if count>=n_filter:
                filtered = filter_words(filtered,gram)
    return filtered
# str = 'the end is nigh. mesh keda. the end is nigh'
# grams = count_ngrams([str])
# print_most_frequent(grams)
#
# fu2 = remove_ngrams(tokenize(str),grams)
hypothesis = []
references = []
filtered_pred = []
for i in range(len(preds)):
    label = labels[i]
    pred = preds[i]
    pred = pred.replace('xxxx','')
    # label = label.replace('xxxx','')
    pred = pred.replace('end','')
    pred = pred.replace('"','')
    pred = pred.replace('  ',' ')
    #
    xx = pred.split('.')
    x = [z for z in xx if len(z.split(" "))>1]
    # print(x[-1])
    nwq = x
    nwq = [x[0]]
    for i in range(1,len(x)):
        if x[i] in nwq:
            continue
        nwq.append(x[i])
    if len(nwq)>1:
        if len(nwq[-1].split()) <= 2:
            pred = '. '.join(nwq[:-1])
        else:
            pred = '. '.join(nwq)
    else:
        pred = '. '.join(nwq)

    # grams = count_ngrams([pred])
    # pred = remove_ngrams(tokenize(pred),grams)
    # pred=' '.join(pred)
    filtered_pred.append(pred)

    target_word_list = tokenizer_wrapper.GPT2_format_output(label)
    hypothesis_word_list = tokenizer_wrapper.GPT2_format_output(pred)
    if hypothesis_word_list[-1] == hypothesis_word_list[-1]:
        hypothesis_word_list = hypothesis_word_list[:-1]
    hypothesis.append(hypothesis_word_list)
    references.append([target_word_list])

scores = get_evalutation_scores(hypothesis, references, False)
print("scores: {}".format(scores))

df['prediction'] = filtered_pred

df.to_csv('filtered_preds.csv',index=False)


# remove " and xxxx from labels
# try softmax on tags
# understand length penalty not working
# bad words ids 12343, 1
# better ending criteria