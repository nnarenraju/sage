# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Record of the date each module was last reviewed, in a JSON file.

Filename         =  review.py
Description      =  Code review for Sage

Created on 19/07/2024 at 11:07:25

__author__       =  Narenraju Nagarajan
__copyright__    =  Copyright 2024, Sage
__credits__      =  nnarenraju
__license__      =  GPL-3.0-or-later
__version__      =  0.0.1
__maintainer__   =  nnarenraju
__affiliation__  =  University of Glasgow
__email__        =  nnarenraju@gmail.com
__status__       =  inUsage


Github Repository: NULL

Documentation: NULL

"""

import os
import json
import subprocess

from json.decoder import JSONDecodeError


def get_sage_abspath():
    """
    Top level of the git repository holding the working directory.

    Returns
    -------
    str
        Absolute path, or an empty string outside a git repository.
    """
    git_revparse = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    repo_abspath = git_revparse.stdout.strip("\n")
    return repo_abspath


def set_review_date(parent_name, module_name, last_review_date):
    """
    Record the last review date of a module in the review JSON file.

    The entry replaces everything recorded under ``parent_name``. The
    file path is hard-coded; ``os.path.join`` discards the repository
    root. An empty or invalid file is started anew.

    Parameters
    ----------
    parent_name : str
        Module group, e.g. ``"data"``.
    module_name : str
        Module reviewed, e.g. ``"real_noise"``.
    last_review_date : str or datetime
        Review date, stored as ``str(last_review_date)``.

    Raises
    ------
    FileNotFoundError
        If the review file does not exist.
    """
    repo_abspath = get_sage_abspath()
    review_file = os.path.join(
        repo_abspath, "/home/nnarenraju/Research/sage/sage/utils/review.json"
    )
    try:
        with open(review_file, "r") as jfile:
            data = json.load(jfile)
        foo = {parent_name: {module_name: str(last_review_date)}}
        data.update(foo)
    except JSONDecodeError:
        # Empty or invalid JSON: start a new record
        data = {parent_name: {module_name: str(last_review_date)}}
    with open(review_file, "w") as jfile:
        json.dump(data, jfile)
