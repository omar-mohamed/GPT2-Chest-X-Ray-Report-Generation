import os
import xml.etree.ElementTree as ET
import random
import pandas as pd
from configs import argHandler  # Import the default arguments
import numpy as np
import re

FLAGS = argHandler()
FLAGS.setDefaults()

# read the reports xml files and create the dataset tsv
reports_path = "IU-XRay/reports"

reports = os.listdir(reports_path)

reports.sort()

reports_with_no_image = []
reports_with_empty_sections = []
reports_with_no_impression = []
reports_with_no_findings = []

images_captions = {}
reports_with_images = {}
text_of_reports = {}


def get_new_csv_dictionary():
    return {'Image Index': [],
            'Patient ID': [],
            'Findings': [],
            'Impression': [],
            'Caption': [],
            'Manual Tags': []
            }


all_data_csv_dictionary = get_new_csv_dictionary()
patient_id = 0
dic = {}
manual_tags_dic = {}
automatic_tags_dic = {}
manual_tags_list = []

for report in reports:

    tree = ET.parse(os.path.join(reports_path, report))
    root = tree.getroot()
    img_ids = []
    # find the images of the report
    images = root.findall("parentImage")
    # if there aren't any ignore the report
    if len(images) == 0:
        reports_with_no_image.append(report)
    else:
        sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
        # find impression and findings sections
        for section in sections:
            if section.get("Label") == "FINDINGS":
                findings = section.text
            if section.get("Label") == "IMPRESSION":
                impression = section.text

        if impression is None and findings is None:
            reports_with_empty_sections.append(report)
        else:
            if impression is None:
                reports_with_no_impression.append(report)
                caption = findings
            elif findings is None:
                reports_with_no_findings.append(report)
                caption = impression
            else:
                caption = "\"" + impression + "\n" + findings + "\""

            manual_tags = root.find("MeSH").findall("major")
            # automatic_tags=root.find("MeSH").findall("automatic")
            manual_tags_tmp = []
            for manual_tag in manual_tags:
                manual_tag = manual_tag.text.lower().strip()
                manual_tag = re.split('/|,', manual_tag)
                for word in manual_tag:
                    word = word.strip()
                    if word in manual_tags_dic.keys():
                        manual_tags_dic[word] += 1
                    else:
                        manual_tags_dic[word] = 1
                    manual_tags_tmp.append(word)

            for image in images:
                manual_tags_list.append(manual_tags_tmp)

                images_captions[image.get("id") + ".png"] = caption
                img_ids.append(image.get("id") + ".png")
                all_data_csv_dictionary['Image Index'].append(image.get("id") + ".png")
                all_data_csv_dictionary['Patient ID'].append(patient_id)
                if findings is None:
                    findings = ""
                if impression is None:
                    impression = ""
                all_data_csv_dictionary['Findings'].append('startseq ' + findings + ' endseq')
                all_data_csv_dictionary['Impression'].append('startseq ' + impression + ' endseq')
                all_data_csv_dictionary['Caption'].append('startseq ' + caption + ' endseq')

            reports_with_images[report] = img_ids
            text_of_reports[report] = caption
            patient_id = patient_id + 1

appearance_limit = 25
to_ignore = []

selected_classes = {}
for tags_list in manual_tags_list:
    tags_str = ''
    tags_list = list(set(tags_list))
    for tag in tags_list:
        if manual_tags_dic[tag] > appearance_limit and tag not in to_ignore:
            selected_classes[tag] = manual_tags_dic[tag]
            if tags_str == '':
                tags_str += tag
            else:
                tags_str += ',' + tag
    if tags_str == '':
        tags_str = 'normal'
    all_data_csv_dictionary['Manual Tags'].append(tags_str)

print(selected_classes.keys())


def split_train_test():
    num_test_images = 500
    num_of_images = len(all_data_csv_dictionary['Image Index'])
    test_indices = random.sample(range(0, num_of_images), num_test_images)

    test_csv_dictionary = get_new_csv_dictionary()
    train_csv_dictionary = get_new_csv_dictionary()

    def append_to_csv_dic(csv_dictionary, index):
        csv_dictionary['Image Index'].append(all_data_csv_dictionary['Image Index'][index])
        csv_dictionary['Patient ID'].append(all_data_csv_dictionary['Patient ID'][index])
        csv_dictionary['Findings'].append(all_data_csv_dictionary['Findings'][index])
        csv_dictionary['Impression'].append(all_data_csv_dictionary['Impression'][index])
        csv_dictionary['Caption'].append(all_data_csv_dictionary['Caption'][index])
        csv_dictionary['Manual Tags'].append(all_data_csv_dictionary['Manual Tags'][index])

    for i in range(num_of_images):
        if i in test_indices:
            append_to_csv_dic(test_csv_dictionary, i)
        else:
            append_to_csv_dic(train_csv_dictionary, i)
    return train_csv_dictionary, test_csv_dictionary


train_csv, test_csv = split_train_test()


def save_csv(csv_dictionary, csv_name, just_caption=False):
    if not just_caption:
        df = pd.DataFrame(csv_dictionary)
        df.to_csv(os.path.join("IU-XRay", csv_name), index=False)
    else:
        df = pd.DataFrame({'Caption': csv_dictionary['Caption']})
        df.to_csv(os.path.join("IU-XRay", csv_name), index=False, header=False)


save_csv(all_data_csv_dictionary, "all_data.csv")
save_csv(train_csv, "training_set.csv")
save_csv(test_csv, "testing_set.csv")
