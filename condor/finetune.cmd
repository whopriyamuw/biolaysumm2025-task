executable = finetune.sh
getenv = true
error = condor.error
log = condor.log
output = condor.out
notification = complete
arguments = ""
transfer_executable = false
request_memory = 16000
request_GPUs = 1
Requirements = (Machine == "patas-gn3.ling.washington.edu")
queue