#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Thu Jun  2 18:35:44 2022

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

# IN-BUILT
import logging
import numpy as np
import tracemalloc
from tqdm import tqdm
import multiprocessing as mp

# Global variable module
# Addresses issue of having large data pickled by MP
# This has to be set as class attribute for it to have global scope
# Using a simple global keyword does not work with the MP worker
import data.cfg as cfg


class SetGlobals:
    """
    SetGlobals
        Work-around to set a global data array accessible to MP worker
    
    Parameters
    ----------
    data: np.ndarray/list
        large input data to be iterated on by the MP worker
    
    Attributes
    ----------
    self.data == data
    set_data: function
        Sets the global data variable in cfg.py to current large data
    
    """
    
    def __init__(self, data, function):
        self.data = data
        self.function = function
    
    def set_data(self):
        cfg.foobar = self.data
    
    def set_func(self):
        cfg.function = self.function


class Parallelise:
    """
    Parallelise
        Generic parallelisation toy-code
        Performs single function operation on iterable
        Expects large dataset as input via global variable
        
    Parameters
    ----------
    func: function
        Function that sets the global variable to large dataset
        This has to be set as an attribute to be global to MP worker
    process: function
        Process function that operates on the units of large dataset
    
    Attributes
    ----------
    worker: MP worker function called by mp.Pool
    initiate: Starts the manager, handles clean termination and outputs data
    
    """
    
    def __init__(self, set_gdata, set_gfunc):
        # Set global attrs
        setattr(self, 'foo', set_gdata())
        setattr(self, 'bar', set_gfunc())
        # Set process params
        self.name = "Transform"
        self.num_workers = 6
        self.num_queues = 0
        self.verbose = True
        self.args = ()
        ## Memory Tracer limits
        # Keep this value low for optimal performace
        # If performace is slower than sequential mode, this may a probable cause
        self.max_mem_limit_in_MB = 30.0
    
    def __str__(self):
        msg = 'Transform: MP {}'.format(self.name)
        msg += '\nnum_workers = {}'.format(self.num_workers)
        msg += '\nargs = {}'.format(self.args)
        msg += '\nMaximum limit on MP RAM = {} MB'.format(self.max_mem_limit_in_MB)
        return msg
    
    def worker(self, input_data, out=None, queues=None):
        # Run function through iterable idx
        iargs = (input_data,) + self.args
        datum = cfg.function(*iargs)
        if queues == None:
            # Give all relevant save data to Queue
            data = {'datum': datum, 'kill': False}
            qidx = np.random.randint(low=0, high=len(queues))
            queues[qidx].put(data)
        else:
            out.append(datum)
    
    def listener(self, queue, qout=None):
        # Queue listener function (writing to out ListProxy)
        while True:
            data = queue.get()
            # if processes have ended, kill
            if data['kill']:
                break
            # Appending to out ListProxy
            qout.append(data['datum'])
            
    def initiate(self):
        # Save tmp hdf5
        # Must use Manager queue here, or will not work
        manager = mp.Manager()
        # Initiate queues
        if self.num_queues > 0:
            queues = [manager.Queue() for _ in range(self.num_queues)]
            out = [manager.list() for _ in range(len(queues))]
        else:
            queues = None
            out = manager.List()
        # Empty jobs every iteration
        jobs = []
        # Initialise pool
        pool = mp.Pool(int(self.num_workers + self.num_queues))
        # Put listener to work first (this will wait for data in MP and write to Queue)
        # NOTE: Cannot pass open files to listener
        if self.num_queues > 0:
            watchers = [pool.apply_async(self.listener, (queue, qout)) for queue, qout in zip(queues, out)]
        
        # Set iterable
        iterable = range(len(cfg.foobar))
        
        for idx in iterable:
            # TODO: if num_queues>0, out needs to be passed properly (not None)
            job = pool.apply_async(self.worker, (cfg.foobar[idx], None, queues))
            jobs.append(job)
        
        # Collect results from the workers through the pool result queue
        tracemalloc.start() # memory tracer
        pbar = tqdm(jobs) if self.verbose else jobs
        for job in pbar:
            mem = tracemalloc.get_traced_memory()
            curr_mem = np.round(mem[0]/(1024*1024), 3)
            max_mem = np.round(mem[1]/(1024*1024), 3)
            # Memory sanity check
            if max_mem > self.max_mem_limit_in_MB:
                print('\n\n')
                logging.critical('Large pickled memory {} MB!'.format(max_mem))
                raise ValueError('MP performing at subpar levels due to memory error.')
            
            if self.verbose:
                pbar.set_description('MP {}: RAM curr={} MB, peak={} MB'.format(self.name, curr_mem, max_mem))
            job.get()
        
        # Kill memory tracers
        tracemalloc.stop()
        # Kill the listener when all jobs are complete
        if self.num_queues > 0:
            _ = [queue.put({'kill': True}) for queue in queues]
            output = np.concatenate(tuple([np.array(foo) for foo in out]))
        else:
            output = np.array(out)
        # End the pool processes
        pool.close()
        pool.join()
        
        # Concatenate all qout together and return to caller
        return output
