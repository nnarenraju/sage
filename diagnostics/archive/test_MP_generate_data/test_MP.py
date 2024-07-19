#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Sat Apr  9 21:05:58 2022

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

import multiprocessing as mp
import numpy as np
import time
from tqdm import tqdm

num_iter = 1000000
num_write = 10000000

def worker(arg, q):
    '''stupidly simulates long running process'''
    s = 'this is a test'
    txt = s
    for i in range(num_iter):
        txt += s 
    
    res = {'fn': "testMP_{}.txt".format(arg), 'kill': False}
    q.put(res)
    return res

def listener(q):
    '''listens for messages on the q, writes to file. '''
    while 1:
        m = q.get()
        if m['kill'] == True:
            break
        with open(m['fn'], 'w') as f:
            message = str(["hello"]*num_write)
            f.write(message)
            f.flush()

def main():
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() - 5)
    
    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for i in range(80):
        job = pool.apply_async(worker, (i, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in tqdm(jobs):
        job.get()

    #now we are done, kill the listener
    q.put({'kill': True})
    pool.close()
    pool.join()

def seq(arg):
    s = 'this is a test'
    txt = s
    for i in range(num_iter):
        txt += s
    
    store_path = "testSEQ_{}.txt".format(arg)
    with open(store_path, 'w') as f:
        message = str(["hello"]*num_write)
        f.write(message)

def run_seq():
    for i in tqdm(range(80)):
        seq(i)

if __name__ == "__main__":
    start = time.time()
    main()
    end_1 = time.time() - start
    print("Time taken MPQ = {}".format(end_1))
    
    start = time.time()
    run_seq()
    end_2 = time.time() - start
    print("Time taken seq = {}".format(end_2))
    
    print("\n speedup = {}".format(end_2/end_1))
