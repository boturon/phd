import tensorflow as tf
import tensorflow.contrib.learn as learn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as sm
import tensorflow.contrib.rnn as rnn
from fedl.utils import BaselineLogger
loggo=BaselineLogger('loggo')
ld=loggo.ld





def model_def(features,targets,mode,params):
    drop_out_prob=float(0.9)
    if mode==learn.ModeKeys.TRAIN:
        drop_out_prob = float(0.9)
        maybedropout=is_training=True
    else:
        drop_out_prob=1
        maybedropout=is_training=False
    normalizer_params={'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with sm.arg_scope([sm.fully_connected, sm.conv2d],
                      normalizer_fn=sm.batch_norm,
                      normalizer_params= normalizer_params,
                      weights_regularizer=sm.l2_regularizer(float(1e-07)),
                      weights_initializer=layers.xavier_initializer()

                      ) as asc:
        pass

    def conv_qd(net,reps,filter_size,output_depth=None,stack_size=2):
        with sm.arg_scope(asc):
            origin_shape=net.get_shape()[3]
            origin= sm.conv2d(net,filter_size, [1,1],padding='SAME') if  abs(int(origin_shape) - filter_size)>0 else net       
            for i in range(reps):
                for j in range(stack_size):
                    net=sm.conv2d(net,filter_size,[1,3],padding='SAME')
                    origin=net=net+origin
        return net
    
    def fc_stack(net,arg_scope=None,reps=3,size=100):
        with sm.arg_scope(asc):
            origin = net
            origin = sm.fully_connected(net,size,activation_fn=None)
            for j in range(reps):
                net = sm.dropout(net, keep_prob=drop_out_prob, is_training=maybedropout)
                net = sm.fully_connected(net,size)
            net=origin+net
        return net

    def ma(net):
        net=tf.nn.avg_pool(net,padding="SAME",ksize=[1,1,3,1],strides=[1,1,1,1])
        net=sm.flatten(net)
        return net

        
    # qdnet=tf.concat([features['GDPnominal'],features['GDPinput'],features['GDPfd'],features['GDPsd'],features['qdates']],axis=3)
    # frednet=tf.concat([features["UNEMPLOYMENT_INPUT"],features["UNEMPLOYMENT_FD"],features["UNEMPLOYMENT_SD"]],axis=3)
    #import tensorflow.contrib.rnn as rnn
    #import tensorflow as tf
    
    
    with tf.name_scope("input") as sc:
        frednet=tf.concat([features["UNEMPLOYMENT_INPUT"],features["UNEMPLOYMENT_FD"],features['UNEMPLOYMENT_SD'],features["UNEMPLOYMENT_Q_INPUT"]],axis=3)
        # frednet=tf.unstack(tf.transpose(tf.squeeze(frednet),[1,0,2]))
        frednet=tf.squeeze(frednet)
        # qdnet = tf.concat([features["UNEMPLOYMENT_Q_INPUT"]], axis=3)
        # qdnet = tf.reshape(qdnet, [-1, 36,1])
    
    with tf.variable_scope("lstm1") as sc:
        mk_lstm = lambda x: [rnn.LSTMCell(5, use_peepholes=True,
                        initializer=layers.xavier_initializer()) for i in range(x)]
        #lstm=rnn.LSTMCell(5,use_peepholes=True,initializer=layers.xavier_initializer())
        lstm=rnn.MultiRNNCell(mk_lstm(3))
        frednet,_=tf.nn.dynamic_rnn(lstm,frednet,dtype=tf.float32)
        
    # with tf.variable_scope('lstm2') as sc:
    #     lstm=rnn.LSTMCell(5,use_peepholes=True,initializer=layers.xavier_initializer())
    #     lstm=rnn.MultiRNNCell([lstm]*3)
        
    #     qdnet,_=tf.nn.dynamic_rnn(lstm,qdnet,dtype=tf.float32)
    #     qdnet=sm.flatten(qdnet)
    
    
    net=sm.flatten(frednet)
    # net=tf.concat([net,qdnet],axis=1)
    
    
    llayer=int(18)
    
    
    net=fc_stack(net,asc,3,llayer)
    
    for i in range(int(2)):
        if int(llayer/2)>0:
            llayer=int(llayer/2)
            net=fc_stack(net,asc,3,llayer)
        else:
            break
    
    net=sm.fully_connected(net,num_outputs=1)
    
    return net
    



lrate = 1e-3
