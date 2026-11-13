"""Example HTCondor submission of a one-GPU job; submits on import."""

import htcondor
import classad

hostname_job = htcondor.Submit({
    "executable": "",               # program run on the execute node
    "arguments": "",                # its arguments
    "request_GPUs": "1",
    "cuda_version": "11.2",
    "output": "test.out",           # standard output
    "error": "test.err",            # standard error
    "log": "test.log",              # job event log
    "request_cpus": "16",
    "request_memory": "10000MB",
    "request_disk": "1000MB",
})

print(hostname_job)

schedd = htcondor.Schedd()
submit_result = schedd.submit(hostname_job)
print(submit_result.cluster())                  # ClusterId
