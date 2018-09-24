
import tensorflow as tf
import fedl
import numpy as np
import tensorflow.contrib.learn as learn
import tensorflow.contrib.slim as sm
import tensorflow.contrib.metrics as metrics
import sys,time,multiprocessing
import time
import os
from fedl.read_records import make_batch_queue,getspec,ordered_batch
from attrdict import AttrDict as ad
from fedl import utils
import glob,sys,os

loggo=utils.BaselineLogger('loggo')
ld=loggo.ld

import argparse
tf.logging.set_verbosity(tf.logging.INFO)

parser=argparse.ArgumentParser()
parser.add_argument("--run_name",default="./template",help="template folder location",dest='template')
parser.add_argument("--subdir",default="grid/",help="path to model_def and test/train.tfrecord",dest='subdir')
parser.add_argument("--epochs",default=100,help='number of epochs to run')

parser.add_argument("--model",default=None,help='non-functioning_argument')
parser.add_argument("--train_file",default=None,help='non-functioning_argument')
parser.add_argument("--test_file",default=None,help='non-functioning_argument')
parser.add_argument("--dataspec",default=None,help='non-functioning_argument')
parser.add_argument("--output_prefix",default=None,help='non-functioning_argument')

parser.add_argument("--target",default='GDP',dest='target')

args = parser.parse_args()
ld(args.subdir)
def joiner(x):
    return os.path.join(args.subdir,x)



#<editor-fold desc="load in model function">
def model_def(features,targets,mode,params):
    pass
with open(joiner('model_def.py'),'r') as f:
    exec(f.read())
#</editor-fold>


# ------------------------------------------ #

name = args.subdir
log_path = './logs/'+name

# ------------------------------------------ #

PARAMS=ad({"train_file":joiner("train.tfrecord"),
        "validate_file":joiner("test.tfrecord"),
          "batch_size":48,"dataspec_file":joiner("dataspec.yaml")})

PARAMS.spec=getspec(PARAMS['dataspec_file'])
PARAMS.model_dir=log_path

#<editor-fold desc="data loading functions">
def train_data_fn(PARAMS):
    def input_fn():
        queue=ordered_batch( filenames=[PARAMS['train_file']],dataspec=PARAMS['spec'])
        #queue=make_batch_queue(PARAMS['batch_size'], filenames=[PARAMS['train_file']],
         #                dataspec=PARAMS['spec'])
        target=queue[args.target]
        return queue, target
    return input_fn

def test_data_fn(PARAMS):
    def input_fn():
        queue=ordered_batch(filenames=[PARAMS['validate_file']],
                         dataspec=PARAMS['spec'])
        target = queue[args.target]
        return queue, target
    return input_fn
# </editor-fold>

#<editor-fold desc="model function">


def model_fn(features,targets,mode,params):
    
    net=model_def(features,targets,mode,params)
    prediction=net
    output_dict={'yhat':prediction}
    #</editor-fold>
    #<editor-fold desc="losses and stuff">
    loss=tf.losses.mean_squared_error(targets,prediction)
    total_loss=tf.losses.get_total_loss()
    # rmsprop=tf.train.RMSPropOptimizer(params['learning_rate'])
    adam=tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = sm.learning.create_train_op(total_loss, adam,summarize_gradients=False,clip_gradient_norm=2)
    #tf.contrib.layers.optimize_loss(
     # loss=total_loss,
     # global_step=tf.contrib.framework.get_global_step(),
     # learning_rate=params.get('learning_rate',.1),
     # optimizer="Adam",
     # summaries=["loss", "learning_rate"])
    eval_metric_ops={"mae":tf.metrics.mean_absolute_error(targets,prediction)}
    tf.summary.scalar("train_loss",loss)
    # </editor-fold>
    return learn.ModelFnOps(
        mode=mode,
        predictions=output_dict,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


# </editor-fold>

#<editor-fold desc="estimation loop">

#tf_random_seed=None
esto=learn.Estimator(model_fn,PARAMS.model_dir,params={'learning_rate':lrate},config=learn.RunConfig(save_checkpoints_secs=None,save_summary_steps=1000,save_checkpoints_steps=1000))

train_input_fn=train_data_fn(PARAMS)
test_input_fn =test_data_fn(PARAMS)
P1=PARAMS.copy()
P1['validate_file']=joiner("validate.tfrecord")
P2=PARAMS.copy()
P2['validate_file']=joiner("train.tfrecord")


vmetrics={'smae':learn.MetricSpec(metric_fn=metrics.streaming_mean_absolute_error,prediction_key='yhat')}
# read https://github.com/tensorflow/tensorflow/issues/6727 when changing this:
vmonitor=learn.monitors.ValidationMonitor(input_fn=lambda: test_input_fn(), metrics=vmetrics, eval_steps=1,every_n_steps=1000,
    early_stopping_metric="smae",
    early_stopping_metric_minimize=True,
    early_stopping_rounds=50000)
if bool(args.epochs):
    esto.fit(input_fn=train_input_fn,steps=int(args.epochs)*1,monitors=[vmonitor])

predict=esto.predict(input_fn=test_data_fn(P1),as_iterable=False)

#
# for i in range(int(args.epochs)):
#      esto.fit(input_fn=train_input_fn,steps=100)
#      ld("beginning eval")
#      ee=esto.evaluate(input_fn=test_input_fn,steps=1)
#      ld(ee)
#
#      if ee['mae']<=np.float32(args.early_stop):
#         break
# predict=esto.predict(input_fn=test_data_fn(PARAMS),as_iterable=False)
def make_outputs(file, outfile,params=PARAMS):
    p1=params.copy()
    p1['validate_file']=joiner(file)
    predict=esto.predict(input_fn=test_data_fn(p1),as_iterable=False)
    np.savetxt(joiner(outfile+".preds"),predict['yhat'])
    test=test_data_fn(p1)
    batch,_=test()
    sess=tf.Session()
    init=[tf.global_variables_initializer(),tf.local_variables_initializer()]
    tf.train.start_queue_runners(sess=sess)
    sess.run(init)
    targets=sess.run(batch)
    ld(targets[args.target])
    ld(predict['yhat'])
    np.savetxt(joiner(outfile+".targets"),targets[args.target])    

make_outputs('validate.tfrecord','experiment')

make_outputs('train.tfrecord','trained_experiment')
