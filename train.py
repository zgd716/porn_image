"""ResNet Train/Eval module.
"""
import time
import six
import sys
import numpy as np
from model import *
from  preprocess import *
import tensorflow as tf
from freeze import *
# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_porn_data_path','data/porn','Filepattern for training data.')# 训练数据路径
tf.app.flags.DEFINE_string('train_unporn_data_path','data/unporn','Filepattern for training data.')# 训练数据路径
tf.app.flags.DEFINE_string('eval_data_path','data/cifar-10-batches-bin/test_batch.bin','Filepattern for eval data')# 测试数据路径
tf.app.flags.DEFINE_string('log_root','temp', 'Directory to keep the checkpoints. Should be a parent directory of FLAGS.train_dir/eval_dir.')# 模型存储路径
tf.app.flags.DEFINE_integer('num_gpus', 0,'Number of gpus used for training. (0 or 1)')# GPU设备数量（0代表CPU）
#tf.app.flags.DEFINE_float('percent_train',0.9,'')


#super params
tf.app.flags.DEFINE_integer('epoches', 3, 'Number of epoch' )
tf.app.flags.DEFINE_integer('batch_size',16,'number of batch_size')
tf.app.flags.DEFINE_float('learning_rate',0.0001,'')





def train():
    # 构建残差网络模型
    model = OpenNsfwModel()
    model.build()  # todo
    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss,global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Prepocess = Preprocess_Image(sess, FLAGS.train_porn_data_path, FLAGS.train_unporn_data_path, FLAGS.batch_size)
        for epoch in range(FLAGS.epoches):
            batches = Prepocess.get_batch()
            for x_batch,y_batch in batches:
                print('the shape of x_batch:{}'.format(x_batch.shape))
                print('the shape of y_batch:{}'.format(y_batch.shape))
                feed_dict = {model.input: x_batch,model.targets:y_batch}
                loss,step,_=sess.run([model.loss,global_step,optimizer],feed_dict)
                print('epoch:{}\tstep:{}\tloss:{}'.format(epoch,step,loss))

        saver = tf.train.Saver()
        saver.save(sess,'ckpt/nsfw.ckpt')
        freeze_graph('ckpt/', 'nsfw_freeze.pb', 'input,predictions')







def main(_):
        train()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()