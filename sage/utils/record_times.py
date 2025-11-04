#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Fri May 20 11:53:22 2022

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
import os 
import datetime
import numpy as np
from collections import defaultdict

# Plotting
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '32'


def plot_split_times(ax, jobs, values, names, tot, errors=None):
    
    left = 0.0
    
    for name, val in zip(names, values):
        
        if name == 'Avg Total Time' or name == 'Total Time' or name == 'Avg Total Time (MP)':
            color = 'gray'
            alpha = 0.4
        else:
            color = None
            alpha = 0.9
        
        # Set label with necessary params
        if name == 'Avg Total Time (MP)':
            label = name+' ({}s)'.format(round(val, 6))
            val = 1e-5
        elif val > 0.01*tot and name != 'Avg Total Time' and name != 'Total Time':
            label = name+' ({}s, {}%)'.format(round(val, 6), round((val/tot)*100.0, 1))
        else:
            if name == 'Avg Total Time' or name == 'Total Time':
                label = name+' ({}s)'.format(round(tot, 6))
            else:
                label = name+' ({}s)'.format(round(val, 8))
            
        ax.barh(jobs, [val], align='center', left=left, 
                height=0.25, label=label,
                capsize=10, alpha=alpha, color=color)
        left += val
    

def _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos):
    ax = fig.add_subplot(plot_num)
    jobs = [job_name]
    # Values to pass to horizontal stacked bar chart
    values = list(times_dict.values())
    sjobs = list(times_dict.keys())
    # Plot stacked bar chart
    plot_split_times(ax, jobs, values, sjobs, total_time)
    
    # Plotting params
    ax.set_yticks(jobs)
    ax.set_xlabel('Time (seconds)')
    ax.grid(True)
    ax.legend(bbox_to_anchor = (1.02, legend_ypos), loc='upper left')
    

def _avg_dicts(load_split_times):
    # Average time taken for load time split
    avg_split = defaultdict(list)
    std_split = defaultdict(list)
    for foo in load_split_times:
        for key, value in foo.items():
            avg_split[key].append(list(value.numpy()))
    # Average all times for split
    for key, values in avg_split.items():
        avg_split[key] = np.mean(values)
        std_split[key] = np.std(values)
    return avg_split, std_split

def record(plot_times, all_total_time, cfg):
    
    # Get all plotting sections
    load_times = plot_times['load']
    train_times = plot_times['train']
    load_split_times = plot_times['section']
    signal_aug = plot_times['signal_aug']
    noise_aug = plot_times['noise_aug']
    transforms = plot_times['transforms']


    # Average time taken to load data
    _load_times = np.array(load_times[1:])-np.array(load_times[:-1])
    # Average time taken to train on loaded data
    _train_times = np.array(train_times[1:])-np.array(train_times[:-1])
    # Average total times
    avg_ltime = round(np.mean(_load_times), 3)
    avg_ttime = round(np.mean(_train_times), 6)
    avg_per_sample = round(np.mean(_load_times)/cfg.batch_size, 3)
    # Get average splits
    avg_split, std_split = _avg_dicts(load_split_times)
    # Append average total time to the dict if needed
    if sum(avg_split.values()) <= avg_per_sample:
        avg_split['Avg Total Time'] = avg_per_sample - sum(avg_split.values())
        std_split['Avg Total Time'] = np.std(_load_times/cfg.batch_size)
    elif sum(avg_split.values()) > avg_per_sample and cfg.num_workers > 0:
        avg_split['Avg Total Time (MP)'] = avg_per_sample
        avg_per_sample = sum(avg_split.values()) - avg_per_sample
    
    
    ## Plotting all section charts
    fig = plt.figure(figsize=(29.0, 19.0))
    title = 'Time-Split (Num Workers={}, Batch={}, Mean Batch Load Time={} s, Mean Batch Train Time={} s)'
    fig.suptitle(title.format(cfg.num_workers, cfg.batch_size, avg_ltime, avg_ttime), y=0.99)
    
    
    ## Total Train Time chart
    train_total = {}
    train_total['Loading Time'] = sum(_load_times)
    train_total['Training Time'] = sum(_train_times)
    train_total['Total Time'] = all_total_time - (sum(_train_times) + sum(_load_times))
    assert all_total_time >= sum(_train_times) + sum(_load_times)
    
    times_dict = train_total
    total_time = all_total_time
    plot_num = 511
    job_name = 'Epoch Training'
    legend_ypos = 0.75
    _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos)
    
    
    ## Section Chart
    times_dict = avg_split
    total_time = avg_per_sample
    plot_num = 512
    job_name = 'Per Sample'
    legend_ypos = 0.98
    _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos)
    
    
    ## Signal Augmentation chart
    avg_split, std_split = _avg_dicts(signal_aug)
    times_dict = avg_split
    total_time = times_dict['Total Time']
    times_dict['Total Time'] = times_dict['Total Time'] - (sum(times_dict.values())-times_dict['Total Time'])
    plot_num = 513
    job_name = 'Signal Augmentation'
    legend_ypos = 0.75
    _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos)
    
    
    ## Noise Augmentation chart
    avg_split, std_split = _avg_dicts(noise_aug)
    times_dict = avg_split
    total_time = times_dict['Total Time']
    times_dict['Total Time'] = times_dict['Total Time'] - (sum(times_dict.values())-times_dict['Total Time'])
    plot_num = 514
    job_name = 'Noise Augmentation'
    legend_ypos = 0.75
    _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos)
    
    
    ## Transforms chart
    avg_split, std_split = _avg_dicts(transforms)
    times_dict = avg_split
    total_time = times_dict['Total Time']
    times_dict['Total Time'] = times_dict['Total Time'] - (sum(times_dict.values())-times_dict['Total Time'])
    plot_num = 515
    job_name = 'Transformations'
    legend_ypos = 0.85
    _plot(fig, times_dict, total_time, plot_num, job_name, legend_ypos)
    
    
    plt.tight_layout()
    filename = 'time_split_batch_{}_num_workers_{}_{}.png'
    fmt_name = filename.format(cfg.batch_size, cfg.num_workers, str(datetime.date.today()))
    plt.savefig(os.path.join(cfg.export_dir, fmt_name))
    plt.close()
