# -*- coding: utf-8 -*-
import tensorflow as tf
from models.rnnModel import *
from models.embedding_utils import *
from sklearn import metrics
import os
import time
from datetime import timedelta
MODE='test'
save_dir='checkpoints/textrnn'
save_path=os.path.join(save_dir,'best_validation') #最佳val结果保存路径

def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

def feed_data(x_batch,y_batch,keep_prob):
    feed_dict={
        model.input:x_batch,
        model.label:y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def evaluate(sess,x_,y_):
    data_len=len(x_)
    eval_batch=batch_iter(x_,y_,config.batch_size)
    total_loss=0.0
    total_acc=0.0
    for x_batch,y_batch in eval_batch:
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch,1.0)
        loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        total_loss+=loss*batch_len
        total_acc+=acc*batch_len
    return total_loss/data_len,total_acc/data_len

def train():
    tensorboard_dir='tensorboard/textrnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss',model.loss)
    tf.summary.scalar('accuracy',model.acc)
    merged_summary=tf.summary.merge_all()
    writer=tf.summary.FileWriter(tensorboard_dir)

    saver=tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True
    with tf.Session(config=Config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(model.embedding,embedding_matrix))
        writer.add_graph(sess.graph)
        print('training and evaluating...')
        start_time=time.time()
        total_batch=0
        best_eval_acc=0.0
        last_improved=0
        required_improvement=5000 #超过1000轮未提升则提前结束训练
        flag=False
        for epoch in range(config.num_epochs):
            print('Epoch:',epoch+1)
            train_batch=batch_iter(train_inputs,train_labels,config.batch_size)
            for x_batch,y_batch in train_batch:
                feed_dict=feed_data(x_batch,y_batch,config.dropout_keep_prob)
                if total_batch%config.save_per_batch==0:
                    s=sess.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,total_batch)
                if total_batch%config.print_per_batch==0:
                    loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
                    loss_val,acc_val=evaluate(sess,val_inputs,val_labels)
                    if acc_val>best_eval_acc:
                        best_eval_acc=acc_val
                        last_improved=total_batch
                        saver.save(sess,save_path)
                        improved_str='*'
                    else:
                        improved_str=''
                    time_dif=get_time_dif(start_time)
                    msg='Iter:{0:>6}, Train Loss:{1:>6.2}, Train Acc:{2:>7.2%},' \
                        'Val Loss:{3:>6.2}, Val Acc:{4:>7.2%}, Time:{5} {6}'
                    print(msg.format(total_batch,loss,acc,loss_val,acc_val,time_dif,improved_str))
                sess.run(model.optimizer,feed_dict=feed_dict)
                total_batch+=1

                if total_batch-last_improved>required_improvement:
                    print('No optimization dor a long time, stop training...')
                    flag=True
                    break
            if flag:
                break

def test():
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True
    with tf.Session(config=Config) as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.restore(sess,save_path)
        test_loss,test_acc=evaluate(sess,test_inputs,test_labels)
        num_batch=int((len(test_inputs)-1)/config.batch_size)+1
        msg='Test Loss:{0:>6.2}, Test Acc:{1:>7.2%}'
        print(msg.format(test_loss,test_acc))
        # y_test_cls=test_labels
        # y_pred_cls=np.zeros(shape=[len(test_inputs),config.num_classes],dtype=np.int32)
        # for i in range(num_batch):
        #     start_id=i*num_batch
        #     end_id=min((i+1)*config.batch_size,len(test_inputs))
        #     feed_dict={
        #         model.input:test_inputs[start_id:end_id],
        #         model.keep_prob:1.0
        #     }
            # y_pred_cls[start_id:end_id]=sess.run(model.predict_class,feed_dict)


if __name__ == '__main__':
    config=RNNConfig()
    train_inputs,train_labels,val_inputs,val_labels,test_inputs,test_labels,embedding_matrix,embedding_word_dict=\
        load_data('../data/train.csv','../data/test.csv','../data/test_labels.csv',500,'../data/crawl-300d-2M.vec')
    # vocab_dir='../data/vocab.txt'
    # words,word_to_id=read_vocab(vocab_dir)
    model=TextRNN(config)
    # _,word_to_id=read_vocab(vocab_dir)
    # train_inputs,train_labels,val_inputs,val_labels,test_inputs,test_labels=raw_data(word_to_id,config.seq_length)

    if MODE=='train':
        train()
    else:
        test()
