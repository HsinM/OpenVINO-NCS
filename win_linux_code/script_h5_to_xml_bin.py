#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input_model_name", help="Input model Path")
#parser.add_argument("input_model_shape", help="Input model Path")
#parser.add_argument("-o", "--optional-arg", help="optional argument", dest="opt", default="default")
args = parser.parse_args()
print("positional arg:", args.input_model_name)
#print("positional arg:", args.input_model_shape)
#print("optional arg:", args.opt)

# In[1]:


from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.python.framework import graph_io


# In[4]:


keras_model = 'best_model.h5'

output_graph_dir = './pb_model'
output_graph_name = keras_model.split('.')[0] + '.pb'


# In[5]:


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
                
        #print('{}:{}\n{}:{}\n{}:{}'.format(len(freeze_var_names), freeze_var_names, 
        #                                   len(freeze_var_names), output_names, 
        #                                   len(freeze_var_names), freeze_var_names))
        
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# In[8]:


h5_model = load_model(keras_model)

frozen = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in h5_model.outputs])

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(frozen)
    
#for op in graph.get_operations():
#    print(op.name)

pb_path = graph_io.write_graph(frozen, output_graph_dir, output_graph_name, as_text=False)

print('freezed pb model saved!!')
pb_path

in_shape = list(h5_model.layers[0].input_shape)
in_shape[0] = 1
#in_shape


# # Convert pb model to IR model using BASH

# In[10]:


import subprocess

commands = '''
. /opt/intel/openvino/bin/setupvars.sh
mo.py --input_model {} --output_dir ./vino_model --input_shape [{}] --data_type=FP16
'''.format(pb_path, ','.join(str(e) for e in in_shape))

process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out, err = process.communicate(commands.encode())

print(out.decode())