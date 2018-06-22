# -*- coding: utf-8 -*-
import tensorflow as tf
class RNNConfig(object):
    embedding_size=300
    num_classes=6
    seq_length=500
    vocab_size=143882
    num_layer=1
    hidden_size=128
    rnn='gru'
    input_dropout_prob=0.8
    dropout_keep_prob=1.0
    learning_rate=1e-3
    batch_size=128
    num_epochs=50
    print_per_batch=100
    save_per_batch=10

class TextRNN(object):
    def __init__(self,config):
        self.config=config
        self.input=tf.placeholder(tf.int32,[self.config.batch_size,config.seq_length],name='input')
        self.label=tf.placeholder(tf.float32,[self.config.batch_size,self.config.num_classes])
        self.keep_prob=tf.placeholder(tf.float32,name='keep_prob')
        self.embedding=tf.get_variable('embedding',[self.config.vocab_size,self.config.embedding_size],trainable=False)
        self.rnn()
    def rnn(self):
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size,state_is_tuple=True)
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_size)
        def drop_out() :
            if self.config.rnn=='lstm':
                cell=lstm_cell()
            else:
                cell=gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.config.dropout_keep_prob)
        with tf.device('/cpu:0'):
            embedding_inputs=tf.nn.embedding_lookup(self.embedding,self.input)
            embedding_inputs=tf.nn.dropout(embedding_inputs,self.config.input_dropout_prob)
        with tf.name_scope('rnn'):
            # cells=[drop_out() for _ in range(self.config.num_layer)]
            # rnn_cell=tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
            # outputs,_=tf.nn.dynamic_rnn(rnn_cell,embedding_inputs,dtype=tf.float32)
            # last=outputs[:,-1,:]#取最后一个timestep的输出作为结果

            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size)

            init_fw = lstm_fw_cell.zero_state(self.config.batch_size, dtype=tf.float32)
            init_bw = lstm_bw_cell.zero_state(self.config.batch_size, dtype=tf.float32)

            weights = tf.get_variable("weights", [2 * self.config.hidden_size, self.config.num_classes], dtype=tf.float32,   #注意这里的维度
                                      initializer = tf.random_normal_initializer(mean=0, stddev=1))
            biases = tf.get_variable("biases", [self.config.num_classes], dtype=tf.float32,
                                     initializer = tf.random_normal_initializer(mean=0, stddev=1))

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                    lstm_bw_cell,
                                                                    embedding_inputs,
                                                                    initial_state_fw = init_fw,
                                                                    initial_state_bw = init_bw)

            outputs = tf.concat(outputs, 2)   #将前向和后向的状态连接起来
            state_out = tf.matmul(tf.reshape(outputs, [-1, 2 * self.config.hidden_size]), weights) + biases  #注意这里的维度
            logits = tf.reshape(state_out, [self.config.batch_size, self.config.seq_length, self.config.num_classes])
            last=logits[:,-1,:]



        with tf.name_scope('score'):
            #全连接层
            fc=tf.layers.dense(last,self.config.hidden_size,name='fc1')
            fc=tf.contrib.layers.dropout(fc,self.keep_prob)
            fc=tf.nn.relu(fc)
            #输出层
            self.logits=tf.layers.dense(fc,self.config.num_classes,name='fc2')
            self.predict_class=tf.cast(tf.greater_equal(tf.sigmoid(self.logits), 0.5), tf.int32)
        with tf.name_scope('optimize'):
            cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.label)
            self.loss=tf.reduce_mean(cross_entropy)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        with tf.name_scope('accuracy'):
            correct= tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.logits), 0.5), tf.int32), tf.cast(self.label, tf.int32))
            self.acc = tf.reduce_mean(tf.reduce_min(tf.cast(correct, tf.float32), 1))

