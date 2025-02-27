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
tmp_dir = os.path.expanduser("~/tmp/")
os.makedirs(tmp_dir, exist_ok=True)

# Now you can safely create the temporary directory
temporary_dir = tempfile.TemporaryDirectory(prefix=tmp_dir)

temporary_dir =  tempfile.TemporaryDirectory(prefix=os.path.expanduser("~/tmp/"))
print("Created temporary directory %s" % temporary_dir)

# Visualization
from tqdm import tqdm


#%%
'''
# Ensure the parent directory exists
tmp_dir = os.path.expanduser("~/tmp/")
os.makedirs(tmp_dir, exist_ok=True)

# Now you can safely create the temporary directory
temporary_dir = tempfile.TemporaryDirectory(prefix=tmp_dir)
data_dir = os.path.expanduser("~/Data/Riksdagen")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

'''
#%% Define urls and download archives
url = lambda y: """https://data.riksdagen.se/dataset/anforande/anforande-%s%s.json.zip""" % (y, str(y+1)[-2:])
rm_names = {"%s/%s" % (y, str(y+1)[-2:]): url(y) for y in range(1993, datetime.today().year-1)}
rm_names['1999/00'] = """https://data.riksdagen.se/dataset/anforande/anforande-19992000.json.zip"""


rm_names = {k: (rm_names[k], os.path.join("/mnt/c/Users/User/thesis/data_import/temp", rm_names[k].split("/")[-1])) for k in rm_names.keys()}

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
db_path = os.path.join("/mnt/c/Users/User/thesis/data_import", "dataset.csv")
df.to_csv(db_path)
