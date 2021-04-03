import pandas as pd
from configs import argHandler
from shutil import copyfile

FLAGS = argHandler()
FLAGS.setDefaults()

df = pd.read_csv(FLAGS.all_data_csv)
captions = df['Caption']

clear_keywords = ['no finding', 'clear', 'stable']
normal_count = 0
for caption in captions:
    for keyword in clear_keywords:
        if keyword in caption:
            normal_count += 1
            break

print(f"normal cases: {normal_count}\nanomaly cases: {len(captions) - normal_count}")
