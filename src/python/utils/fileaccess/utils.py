import csv
import os
import pickle
from pprint import pprint
import yaml


def load_file(filename: str):
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            return f.read()
    elif filename.endswith('.yml'):
        with open(filename, 'r') as f:
            return yaml.load(f)
    elif filename.endswith('.csv'):
        rows = []
        with open(filename, 'r') as f:
            for row in csv.reader(f):
                rows.append(row)
        return rows
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)


def save_file(file, name: str, path: str = './', verbose=True,raw=False):
    if not os.path.exists(path):
        os.makedirs(path)

    if name.endswith('.pkl'):
        with open(path + name, 'wb') as f:
            pickle.dump(file, f)
    elif name.endswith('.txt'):
        with open(path + name, 'w') as f:
            if raw:
                f.write(file)
            else:
                pprint(file, stream=f)
    elif name.endswith('.yml'):
        with open(path + name, 'w') as f:
            yaml.dump(file)
    elif name.endswith('.cfg'):
        with open(path + name,'w') as f:
            f.write(file)
    else:
        name += '.pkl'
        with open(path + name, 'wb') as f:
            pickle.dump(file, f)

    if verbose:
        print("File stored at: " + path + name)


def create_dirs(dir_names: [str]):
    for dir_name in dir_names:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
