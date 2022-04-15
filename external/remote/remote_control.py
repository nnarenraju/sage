# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Filename        = Foobar.py
Description     = Lorem ipsum dolor sit amet

Created on Tue Jan 18 09:42:29 2022

__author__      = nnarenraju
__copyright__   = Copyright 2021, ProjectName
__credits__     = nnarenraju
__license__     = MIT Licence
__version__     = 0.0.1
__maintainer__  = nnarenraju
__email__       = nnarenraju@gmail.com
__status__      = ['inProgress', 'Archived', 'inUsage', 'Debugging']


Github Repository: NULL

Documentation: NULL

"""

import base64
import paramiko

# Details
server = "wiay.astro.gla.ac.uk"
username = "nnarenraju"
# Password (insecure obfuscation)
obfuscated_pass = b'YnJhTmUyMTUx'

# Command
remote_command = "cd ML-MDC1-Glasgow/"

# Running SSH
ssh = paramiko.SSHClient()
# NOTE: AutoAddPolicy is not secure. Use only within private networks.
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
password = base64.b64decode(obfuscated_pass).decode("utf-8")
ssh.connect(hostname=server, port=22, username=username, password=password)
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(remote_command)

print(ssh_stdout.read())
