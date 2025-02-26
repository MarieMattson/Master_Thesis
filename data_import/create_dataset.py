#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the easily avavlible data archives for older 'Riksmöten'. This builds
the database, which should be updated to include the latest data using another
script.

https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anforanden/

@author: Fredrik Wahlberg <fredrik.wahlberg@lingfil.uu.se>
"""


# Web
from urllib.request import urlretrieve

# Database
import pandas as pd
import numpy as np
from datetime import datetime

# File handling
from zipfile import ZipFile
import json
import os.path
import tempfile
temporary_dir =  tempfile.TemporaryDirectory(prefix=os.path.expanduser("~/tmp/"))
print("Created temporary directory %s" % temporary_dir)

# Visualization
from tqdm import tqdm


#%%
data_dir = os.path.expanduser("~/Data/Riksdagen")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)


#%% Define urls and download archives
url = lambda y: """https://data.riksdagen.se/dataset/anforande/anforande-%s%s.json.zip""" % (y, str(y+1)[-2:])
rm_names = {"%s/%s" % (y, str(y+1)[-2:]): url(y) for y in range(1993, datetime.today().year-1)}
rm_names['1999/00'] = """https://data.riksdagen.se/dataset/anforande/anforande-19992000.json.zip"""


rm_names = {k: (rm_names[k], os.path.join(temporary_dir.name, rm_names[k].split("/")[-1])) for k in rm_names.keys()}

for _, (url, fn) in rm_names.items():
    if not os.path.exists(fn):
        urlretrieve(url, fn)
        print("Downloaded", url)


#%% Accumulate data files into one db
data_accumulator = dict() # Accumulating in doct for efficiency
for rm, (_, fn) in list(rm_names.items()):
    with ZipFile(fn, mode='r') as archive:
        for archive_fn in tqdm(archive.namelist(), desc="Reading items from riksmöte %s" % rm):
            # Parse the data
            with archive.open(archive_fn) as file:
                json_data = file.read()
            data_dict = json.loads(json_data)['anforande']
            # Make sure we have all the keys
            for k in data_dict.keys():
                if k not in data_accumulator.keys():
                    data_accumulator[k] = list()
                    current_length = np.max([len(data_accumulator[k]) for k in data_accumulator.keys()])
                    if len(data_accumulator[k]) < current_length:
                        data_accumulator[k].extend([None]*(current_length-len(data_accumulator[k])))
            # Loop over the accumulator keys to ensure equal length of lists
            for k in data_accumulator.keys():
                if k in data_dict.keys():
                    data_accumulator[k].append(data_dict[k])    
                else:
                    data_accumulator[k].append(None)
df = pd.DataFrame.from_dict(data_accumulator)


#%% Cleaning up temporary files
temporary_dir.cleanup()


#%% Store the data
print("Writing JSON to disk")
db_path = os.path.join(data_dir, "dataset.json.gz")
df.to_json(db_path)


----



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This updates the dataset with the latests data using api.riksdagen.se .


@author: Fredrik Wahlberg <fredrik.wahlberg@lingfil.uu.se>
"""

from urllib.request import urlopen
import os
from multiprocessing import Pool

import pandas as pd
import numpy as np
from datetime import datetime
import re
import html
# conda install spacy
# conda update pydantic
# python -m spacy download sv_core_news_sm
import spacy

from tqdm import tqdm


#%% Load the dataset

data_dir = os.path.expanduser("~/Data/Riksdagen")
db_path = os.path.join(data_dir, "dataset.json.gz")
assert os.path.exists(db_path), "Can't find the database"

print("Loading the dataset")
db_path = os.path.join(data_dir, "dataset.json.gz")
dataset = pd.read_json(db_path)


#%% Download metadata for the latest 'Riksmöte'
current_year = datetime.today().year
a = str(current_year-1)
b = str(current_year)[-2:]
url = """https://data.riksdagen.se/anforandelista/?rm=""" + a + """%2F""" + b + """&anftyp=&d=&ts=&parti=&iid=&sz=100000&utformat=xml"""

with urlopen(url) as remote:
    records = pd.read_xml(remote.read())

print("Downloaded a record with %i items." % len(records))


# Check assumptions beforer merging on 'anforande_id'
assert len(records) == len(records['anforande_id'].unique()), "Duplicate items records"
assert len(dataset) == len(dataset['anforande_id'].unique()), "Duplicate items in dataset"
# assert len(set(dataset['anforande_id']).intersection(set(records['anforande_id']))) == 0, "Intersection not empty between records and dataset"


# Filter records by merge_id_set to get only new ones
merge_id_set = set(records['anforande_id']).difference(set(dataset['anforande_id']))
print(" %i items (by 'anforande_id') are not found in the database" % len(merge_id_set))

new_data = records.loc[records['anforande_id'].isin(merge_id_set)]
# assert len(set(dataset.keys()).difference(set(new_data.keys())))==0, "Not all dataset keys in new_data"
dataset = pd.concat((dataset, new_data), ignore_index=True)

# Check if the anforande_id are unique
assert len(set(dataset['anforande_id'])) == len(dataset)


#%% Check data



# dataset['anforandetext'] = dataset['anforandetext'].astype('string')
# dataset['new_data'] = dataset['new_data'].astype('string')

#%% Download 'anförandetext'
nans_to_fix = np.where(pd.isna(dataset['anforandetext']))[0]
print("Found %i items without 'anforandetext'." % len(nans_to_fix))

for index in tqdm(nans_to_fix, desc=" Downloading 'anforandetext'"):
    assert pd.isna(dataset.at[index, 'anforandetext']), "'anforandetext' is not na at %i" % index
    if not pd.isna(dataset.at[index, 'anforande_url_xml']):
        anforande_url = dataset.at[index, 'anforande_url_xml']
        with urlopen(anforande_url) as remote:
            data = remote.read()
        anforandetext = re.findall(r'\<anforandetext\>(.*)\<\/anforandetext\>', data.decode('utf-8'), flags=re.IGNORECASE+re.MULTILINE)[0]
        assert len(anforandetext) > 0
        anforandetext = html.unescape(anforandetext)
        assert len(anforandetext) > 0
        dataset.at[index, 'anforandetext'] = anforandetext


#%% Check if the charset was preserved
# TODO Check this. I think it is, but an assert wouldn't hurt.
# assert


#%% Clean all 'anforandetext', removing html tags, avstavning etc.
for index in tqdm(range(len(dataset)), desc="Cleaning the text"):
    anforandetext = dataset.loc[index, 'anforandetext']
    if type(anforandetext) is str:
        # Place \n\n at end of html paragraphs
        anforandetext = anforandetext.replace("</p>", '\n\n')
        # Remove start of paragraph tags and emphasis
        for tag in ['<p>', '<em>', '</em>', '<strong>', '</strong>']:
            anforandetext = anforandetext.replace(tag, '')
        # Remove "avstavning"
        anforandetext, _ = re.subn(r'(\w+)(-\n)(\w+)', r'\1\3', anforandetext, flags=re.IGNORECASE)#, count=0, flags=0)¶
        # Insert spaces instead of linefeed at ends of lines
        anforandetext, _ = re.subn(r'([\w.,!?:]+)(\n)(\w+)', r'\1 \3', anforandetext, flags=re.IGNORECASE)#, count=0, flags=0)¶
        # Clean starts and ends of text
        anforandetext = anforandetext.strip()
        # Replace the text in the dataset
        dataset.at[index, 'anforandetext'] = anforandetext

#%% Parsing
print("Parsing text")

for key in ['Part-of-Speech', 'Named Entities', 'VERB', 'ADJ']:
    if key not in dataset.keys():
        print(" Creating '%s' column" % key)
        dataset[key] = pd.NA

needs_parsing = np.where(pd.isna(dataset['VERB']) | pd.isna(dataset['ADJ']))[0]
np.random.shuffle(needs_parsing)
# needs_parsing = needs_parsing[:2**16]
# needs_parsing = np.where(~pd.isna(dataset['Named Entities']))[0]
print(" Found %i items without parsing data" % len(needs_parsing))


if len(needs_parsing) > 0:
    nlp = spacy.load("sv_core_news_sm")
    # nlp.select_pipes(disable=['ner', 'tok2vec'])
    print(" Made a spacy pipeline with", [a for a, b in nlp.pipeline])


# for index in tqdm(needs_parsing, desc=" Parsing"):
#     # sents = list()
#     # ents = list()
#     verb = list()
#     adj = list()
#     if type(dataset.at[index, 'anforandetext']) is str:
#         doc = nlp(dataset.at[index, 'anforandetext'])
#         # ents = [(ent.text, ent.label_) for ent in doc.ents]
#         for sent in doc.sents:
#             # sents.append((sent.text, [(token.text, token.pos_) for token in sent]))
#             verb.extend([token.text for token in sent if token.pos_.find("VERB") >= 0])
#             adj.extend([token.text for token in sent if token.pos_.find("ADJ") >= 0])
#         dataset.at[index, 'VERB'] = " ".join(verb)
#         dataset.at[index, 'ADJ'] = " ".join(adj)
#         # dataset.at[index, 'Part-of-Speech'] = sents
#         # dataset.at[index, 'Named Entities'] = ents


# import gc

def parse_anforandetext(data):
    # nlp = spacy.load("sv_core_news_sm")
    index, text = data
    # if np.random.randint(100) == 0:
    #     gc.collect()
    # sents = list()
    # ents = list()
    verb = list()
    adj = list()
    if type(text) is str:
        doc = nlp(text)
        # ents = [(ent.text, ent.label_) for ent in doc.ents]
        for sent in doc.sents:
            # sents.append((sent.text, [(token.text, token.pos_) for token in sent]))
            verb.extend([token.text for token in sent if token.pos_.find("VERB") >= 0])
            adj.extend([token.text for token in sent if token.pos_.find("ADJ") >= 0])
        return (index, {'VERB': " ".join(verb), 'ADJ': " ".join(adj)})
    else:
        return None

def item_iterator(indices):
    for index in indices:
        yield (index, dataset.at[index, 'anforandetext'])

pool = Pool(os.cpu_count()-1)
pbar = tqdm(total=len(needs_parsing), desc=" Parsing")
for data in pool.imap_unordered(parse_anforandetext,
                                item_iterator(needs_parsing),
                                chunksize=100):
    # if np.random.randint(100) == 0:
    #     gc.collect()
    if data is not None:
        index, results = data
        for key in ['VERB', 'ADJ']:#, 'Part-of-Speech', 'Named Entities']:
            dataset.at[index, key] = results[key]
    pbar.update(1)
pbar.close()
pool.close()

    
#%% Parallel pandas 26it/s
# tqdm.pandas(ncols=10)
# pip install --upgrade parallel-pandas
# from parallel_pandas import ParallelPandas
# ParallelPandas.initialize(n_cpu=os.cpu_count(), split_factor=4,
#                           show_vmem=False, disable_pr_bar=True)

# def parse_rows(row):
#     # nlp = spacy.load("sv_core_news_sm")
#     sents = list()
#     ents = list()
#     if type(row['anforandetext']) is str:
#         doc = nlp(row['anforandetext'])
#         ents = [(ent.text, ent.label_) for ent in doc.ents]
#         for sent in doc.sents:
#             sents.append((sent.text, [(token.text, token.pos_) for token in sent]))
#         return {'Part-of-Speech': sents, 'Named Entities': ents}
#     else:
#         return None

# needs_parsing = dataset[pd.isna(dataset['Part-of-Speech']) | pd.isna(dataset['Named Entities'])]
# pbar = tqdm(total=len(needs_parsing))
# n_chunk = 1024*2
# for i in range(len(needs_parsing)//n_chunk+1):
#     needs_parsing = dataset[pd.isna(dataset['Part-of-Speech']) | pd.isna(dataset['Named Entities'])]
#     needs_parsing = needs_parsing[:n_chunk]
#     # needs_parsing = dataset[~pd.isna(dataset['Named Entities'])]
#     parsing_results = needs_parsing.p_apply(parse_rows, axis=1)
#     for index, results in zip(needs_parsing.index, parsing_results):
#         dataset.at[index, 'Part-of-Speech'] = results['Part-of-Speech']
#         dataset.at[index, 'Named Entities'] = results['Named Entities']
#     pbar.update(n_chunk)
# pbar.close()


#%% Download ledamotsdata
people = pd.read_csv("""https://data.riksdagen.se/dataset/person/person.csv.zip""")
print("Found %i items with categories: %s" % (len(people), people.keys()))
print("%i unique ids" % len(set(people['Id'])))


#%% Time in parliament at item time and fix intressent_id
# Find the number of days since first appointment
id2first_appointment = dict()
for i in tqdm(range(len(people)),
              desc='Finding dates of first appointments'):
    t = pd.to_datetime(people.loc[i, 'From'])
    k = str(people.loc[i, 'Id']).strip()
    if k not in id2first_appointment:
        id2first_appointment[k] = t
    elif id2first_appointment[k] > t:
        id2first_appointment[k] = t

dataset['intressent_id'] = [item if item is not None else np.nan for item in dataset['intressent_id']]
print(" Found %i unique intressent_ids" % len(set(dataset['intressent_id'])))
print(" Found %i items without an 'intressent_id'" % np.sum(pd.isna(dataset['intressent_id'])))

print(" Fixing intressent_id")
# Most keys have 13 characters
intressent_ids = list()
for item in dataset['intressent_id']:
    if item in id2first_appointment.keys():
        intressent_ids.append(item)
    elif type(item) is float and not np.isnan(item):
        key = "0"+str(int(item))
        if key in id2first_appointment.keys():
            intressent_ids.append(key)
        else:            
            intressent_ids.append(item)
    else:            
        intressent_ids.append(item)
assert len(dataset) == len(intressent_ids)
dataset['intressent_id'] = intressent_ids

# TODO Look more at the intressent_ids that could not be mapped to a person
remainder = [item for item in intressent_ids if item not in id2first_appointment.keys()]
remainder = [item for item in remainder if not (type(item) is float and np.isnan(item))]
# [type(item) for item in remainder]
print(" %i unique ids in %i items could not be mapped to a person" % (len(set(remainder)), len(remainder)))


#%% Parse dates and days since first appointment
assert len(dataset['dok_datum']) == len(dataset['intressent_id'])
timestamps = list()
for dok_datum, speaker_id in tqdm(zip(dataset['dok_datum'], dataset['intressent_id']),
                      desc="Parsing dok_datum to timestamps",
                      total=len(dataset['dok_datum'])):
    if len(dok_datum) == 19:
        dok_date = datetime.strptime(dok_datum, "%Y-%m-%d %H:%M:%S")
    else:
        dok_date = datetime.strptime(dok_datum, "%Y-%m-%d")
    timestamps.append(pd.Timestamp(dok_date))
dataset['timestamps'] = timestamps
print(" Parsed %i dates" % np.sum(np.invert(pd.isna(timestamps))))


# Create the new column
dataset['days_since_first_appointment'] = np.nan

days_since_first_appointment = list()
for timestamp, speaker_id in tqdm(zip(dataset['timestamps'], dataset['intressent_id']),
                      desc="Find the number of days since first appointment",
                      total=len(dataset['dok_datum'])):
    if speaker_id in id2first_appointment.keys():
        days = (timestamp - id2first_appointment[speaker_id]).days
        days_since_first_appointment.append(days)
    else:
        days_since_first_appointment.append(np.nan)
dataset['days_since_first_appointment'] = days_since_first_appointment

print(" Found days since first appointment for %.1f%% of items" %
      (100-100*np.sum(pd.isna(dataset['days_since_first_appointment']))/len(dataset)))

# TODO Testcases


#%% Something with titles
print("Found %i unique titles belonging to %i people" %
      (len(set(people['Titel'])), len(set(people['Id']))))

id2titles = dict()
title2ids = dict()
for index in tqdm(people.index, desc='Processing titles'):
    # timestamp = pd.to_datetime(people.loc[index, 'From'])
    key = str(people.loc[index, 'Id']).strip()
    title = people.loc[index, 'Titel']
    if type(title) is float and np.isnan(title):
        continue
    title = str(title).strip()
    if key not in id2titles.keys():
        id2titles[key] = [title]
    else:
        id2titles[key].append(title)
    if title not in title2ids.keys():
        title2ids[title] = {key}
    else:
        title2ids[title].add(key)

# [item for item in title2ids.keys()]
print("Found %i representatives (out of %i) without a title" %
      (len(set(people['Id']).difference(id2titles.keys())), len(set(people['Id']))))


# Titles more common in the past? len untitled per decade?
# Geography north/south, stad/landsbyggd, population density of district?

#%% Normalize party code

print("Found %i unique party codes" % len(set(dataset['parti'])))

parties = {'V', 'S', 'MP', 'C', 'L', 'M', 'KD', 'NYD', 'SD'}
normalized_party_names = [name.upper() if type(name) is str else name for name in dataset['parti']]
conversions = {'KDS': 'KD', 'FP': 'L'}
normalized_party_names = [conversions[name] if name in conversions.keys() else name for name in normalized_party_names]
normalized_party_names = [name if name in parties else np.nan for name in normalized_party_names]
dataset['normalized_party'] = normalized_party_names

print(" Made code conversions:", {item for item in zip(dataset['parti'], dataset['normalized_party'])})
print(" %i items in the dataset (%.1f%%) have normalized party codes" %
      (len(dataset)-pd.isna(dataset['normalized_party']).sum(),
       100*(len(dataset)-pd.isna(dataset['normalized_party']).sum())/len(dataset)))


#%% Gender of item speaker

# Data uses two genders, this might change in the future

assert len(set(people['Kön'])) == 2

id2gender = dict()
for i in tqdm(range(len(people)), desc='Matching gender and identifier id'):
    id2gender[people.loc[i, 'Id']] = people.loc[i, 'Kön']

gender = list()
for item in tqdm(dataset['intressent_id'], desc="Mapping gender data to dataset ids"):
    if item in id2gender.keys():
        gender.append(id2gender[item])
    else:
        gender.append(np.nan)
dataset['gender'] = gender


#%% Store the updated database
print("Writing JSON to disk")
db_path = os.path.join(data_dir, "dataset.json.gz")
dataset.to_json(db_path)


#%% Setup Spacy and NLTK
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stopword_list = stopwords.words('swedish')

# Spacy's stop word definition is too wide
# from spacy.lang.sv import stop_words
# stopword_list = list(stop_words.STOP_WORDS)

# Spacy's tokenizer is used later in the pipeline for tagging etc.
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize#, sent_tokenize
# word_tokenize(text, language='swedish', preserve_line=False)
# sent_tokenize(text, language='swedish')
# def custom_tokenizer(text):
#     return word_tokenize(text, language='swedish', preserve_line=False)

# spacy_swedish_tokenizer = spacy.load("sv_core_news_sm")
# spacy_swedish_tokenizer.select_pipes(disable=[name for name, obj in spacy_swedish_tokenizer.pipeline])

# def custom_tokenizer(text):
#     return [token.text for token in spacy_swedish_tokenizer(text)]
#     # return [token.text.lower() for token in spacy_swedish_tokenizer(text)]

# Create count vectors
vectorized_text_path = os.path.join(data_dir, "vectorized_text.npz")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from utils import build_tokenizer_pipeline

print("Creating count vectors...", end="")
# CountVectorizer is not case sensitive by default
vectorizer = CountVectorizer(tokenizer=build_tokenizer_pipeline(),
                             token_pattern=None,
                             stop_words=stopword_list)
X_raw = [anforandetext if type(anforandetext) is str else '' for anforandetext in dataset['anforandetext']]
count_vectors = vectorizer.fit_transform(X_raw)
count_vector_vocab = vectorizer.get_feature_names_out()
count_vector_id = np.asarray(dataset['anforande_id'])
print("done (%iMb)" % (count_vectors.data.nbytes//1e6))

print("Creating tfidf vectors...", end="")
vector_transformer = TfidfTransformer()
tfidf_vectors = vector_transformer.fit_transform(count_vectors)
print("done (%iMb)" % (tfidf_vectors.data.nbytes//1e6))

print("Creating VERB and ADJ count vectors...", end="")
vectorizer = CountVectorizer()
X_raw = [text if type(text) is str else '' for text in dataset['VERB']]
verb_count_vectors = vectorizer.fit_transform(X_raw)
verb_count_vector_vocab = vectorizer.get_feature_names_out()
verb_count_vector_id = np.asarray(dataset['anforande_id'])
X_raw = [text if type(text) is str else '' for text in dataset['ADJ']]
adj_count_vectors = vectorizer.fit_transform(X_raw)
adj_count_vector_vocab = vectorizer.get_feature_names_out()
adj_count_vector_id = np.asarray(dataset['anforande_id'])
print("done (%iMb)" % ((verb_count_vectors.data.nbytes+adj_count_vectors.data.nbytes)//1e6))

# tfidf_vectors.astype('float32').data.nbytes

np.savez(vectorized_text_path,
         count_vectors = count_vectors,
         count_vector_vocab = count_vector_vocab,
         count_vector_id = count_vector_id,
         tfidf_vectors = tfidf_vectors,
         verb_count_vectors = verb_count_vectors,
         verb_count_vector_vocab = verb_count_vector_vocab,
         verb_count_vector_id = verb_count_vector_id,
         adj_count_vectors = adj_count_vectors,
         adj_count_vector_vocab = adj_count_vector_vocab,
         adj_count_vector_id = adj_count_vector_id)