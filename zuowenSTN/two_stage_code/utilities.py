from collections import namedtuple
import json
import os
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage

def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj

def get_config_list(config_list):
  return tuple(config_list)


def log_to_file(logfile, str):
    with open(logfile, "a") as f:
        f.write(str)
        f.flush()


def concatenate_json_files(file_list, filename_metafile):
    if not os.path.isfile(filename_metafile):
        global_dict = {}
    else:
        with open(filename_metafile, 'r') as f:
            global_dict = json.load(f)
    for file in file_list:
        with open(file, 'r') as f:
            experiments_f = json.load(f)
            for key, value in experiments_f.items():
                global_dict[key] = value

    with open(filename_metafile, 'w') as f:
        json.dump(global_dict, f, indent=2, sort_keys=True)

#for sanity check, by default show first 4 == (n^2) images in each batch
def show_image_batch(x_batch, n = 2, title = 'title'):
    pyplot.figure(1)
    k = 0
    for i in range(0, n):
        for j in range(0, n):
            pyplot.subplot2grid((n, n), (i, j))
            pyplot.imshow(toimage(x_batch[k]))
            k = k + 1
    pyplot.title(title)
    pyplot.show()