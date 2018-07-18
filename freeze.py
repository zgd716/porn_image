__author__ = 'jellyzhang'
import os
import tensorflow as tf
from tensorflow.python.framework import graph_util
"""
freeze.py:
        主要用来固化图和模型参数，生成缩减版的pb文件
"""

def  freeze_graph(model_path,out_model_name,output_node_names):
    if os.path.exists(model_path):
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        # with open('name','w')as fwrite:
        #     for nodedef in input_graph_def.node._values:
        #         fwrite.write('{}\n'.format(nodedef.name))
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(",")  # We split on comma for convenience
            )
            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(os.path.join(model_path,out_model_name), "wb") as f:
                f.write(output_graph_def.SerializeToString())
                print('成功转换成固化模型')

    else:
            print('请检查模型文件是否存在')




if __name__=='__main__':
    try:
        freeze_graph('ckpt/','nsfw_freeze.pb','input,predictions')
    except Exception as ex:
        print(ex)