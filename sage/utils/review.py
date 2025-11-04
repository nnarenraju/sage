# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename         =  review.py
Description      =  Code review for Sage

Created on 19/07/2024 at 11:07:25

__author__       =  Narenraju Nagarajan
__copyright__    =  Copyright 2024, Sage
__credits__      =  nnarenraju
__license__      =  MIT Licence
__version__      =  0.0.1
__maintainer__   =  nnarenraju
__affiliation__  =  University of Glasgow
__email__        =  nnarenraju@gmail.com
__status__       =  inUsage


Github Repository: NULL

Documentation: NULL

"""

# Modules
import os
import json
import subprocess

# JSON error
from json.decoder import JSONDecodeError


def get_sage_abspath():
    # Get Sage abspath
    git_revparse = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output = True, text = True)
    repo_abspath = git_revparse.stdout.strip('\n')
    return repo_abspath


def set_review_date(parent_name, module_name, last_review_date):
    repo_abspath = get_sage_abspath()
    review_file = os.path.join(repo_abspath, "utils/review.json")
    try:
        with open(review_file, "r") as jfile:
            data = json.load(jfile)
        foo = {parent_name: {module_name: str(last_review_date)}}
        data.update(foo)
    except JSONDecodeError:
        # in case JSON is empty
        data = {parent_name: {module_name: str(last_review_date)}}
    # Update JSON file
    with open(review_file, "w") as jfile:
        json.dump(data, jfile)