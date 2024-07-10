import htcondor
import classad

hostname_job = htcondor.Submit({
    "executable": "/home/nnarenraju/venv/mlmdc/bin/python3",        # the program to run on the execute node
    "arguments": "/home/nnarenraju/Research/ML-GWSC1-Glasgow/source/baseline/train.py --config KF_BatchTrain --manual",
    #"request_GPUs": "1",
    "cuda_version": "11.2",
    "output": "test.out",       # anything the job prints to standard output will end up in this file
    "error": "test.err",        # anything the job prints to standard error will end up in this file
    "log": "test.log",          # this file will contain a record of what happened to the job
    "request_cpus": "5",            # how many CPU cores we want
    "request_memory": "10000MB",      # how much memory we want
    "request_disk": "1000MB",        # how much disk space we want
})

print(hostname_job)

schedd = htcondor.Schedd()                   # get the Python representation of the scheduler
submit_result = schedd.submit(hostname_job)  # submit the job
print(submit_result.cluster())               # print the job's ClusterId



