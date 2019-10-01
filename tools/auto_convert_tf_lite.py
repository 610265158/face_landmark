
import os
import tensorflow as tf



###caution   it is a little complex

### produce a ckpt file that training_flag in False mode
command_for_make_it_eval="python tools/net_work_for_tf_lite.py"
os.system(command_for_make_it_eval)

print('done step 1')

###  freeze the deployed ckpt
command_for_freeze="python tools/auto_freeze.py"
os.system(command_for_freeze)
print('done step 2')

###  convert to tf_lite
command_for_freeze="python tools/tf_lite.py"
os.system(command_for_freeze)
print('done step 3')
print('over')