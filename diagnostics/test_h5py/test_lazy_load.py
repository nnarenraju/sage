#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Wed Apr 27 12:24:29 2022

__author__      = nnarenraju
__copyright__   = Copyright 2022, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import os
import gc
import sys
import h5py
import time
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})


def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


def make_large_dataset(rows=100, columns=100, name="testfile"):
    with h5py.File("{}.hdf5".format(name), "w") as f:
        dset = f.create_dataset("mydataset", (rows, columns), dtype=np.float32)
        dset[...] = np.random.rand(rows, columns)

def check_lazy_load(name, check_size, dataset=None):
    if dataset!=None:
        start = time.time()
        dset = dataset[:int(check_size)]
        fin = time.time() - start
        return get_obj_size(dset), fin
    else:
        start = time.time()
        with h5py.File('{}.hdf5'.format(name), 'r') as f:
            dset = f['mydataset'][:int(check_size)]
            fin = time.time() - start
            # print('time taken to load data = {} s'.format(fin))
            return get_obj_size(dset), fin
    

""" Create a large dataset and store in hdf5 """
name = 'testfile_chonky_boi_300000'
num_samples = 300000
len_sample = 3715*2

plt.figure(figsize=(9.0, 9.0))
plt.grid(True)

f = h5py.File('{}.hdf5'.format(name), 'r')
dset = f['mydataset']

for _ in range(20):
    load_times = []
    load_size = []
    
    for load_chunk in np.logspace(0.01, np.log10(num_samples), 25, endpoint=True):
        load_chunk = int(load_chunk)
        # Creating a chonky boi dataset if not present
        if not os.path.exists(name+'.hdf5'):
            make_large_dataset(rows=num_samples, columns=len_sample, name=name)
        # Lazy load a part of chonky boi
        obj_size, time_taken = check_lazy_load(name, load_chunk, dset)
        # Display stuff
        # print('size of loaded chunk = {} MB'.format(obj_size/1e6))
        load_times.append(time_taken)
        load_size.append(obj_size/1e6)
    
    plt.plot(load_size, load_times, marker='*')
    plt.yscale('log')
    plt.xscale('log')
    # plt.title('Lazy Loading the Chonky Boi Dataset')
    plt.xlabel('Lazy loaded chunk size (MB)')
    plt.ylabel('Load times (seconds)')

plt.savefig('lazy_loading.png')
plt.show()
plt.close()













