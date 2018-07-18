import tensorflow as tf
import argparse
from image_utils import *
#加载固化模型
def load_pb(frozen_graph_filename):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(frozen_graph_filename, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")


            # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            output_graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph



def predict(img):
    graph = load_pb('ckpt/nsfw_freeze.pb')
    # textcnn  预测
    x = graph.get_tensor_by_name('prefix/input:0')
    #logits =graph.get_tensor_by_name('prefix/logits:0')
    predictions =graph.get_tensor_by_name('prefix/predictions:0')
    with tf.Session(graph=graph) as sess:
        fn_load_image = create_tensorflow_image_loader(sess)
        image = fn_load_image(img)
        pred= sess.run([predictions], feed_dict={x:image})
        print(pred)

if __name__=="__main__":
        parser=argparse.ArgumentParser()
        parser.add_argument('-i','--img',help='图片地址')
        args=parser.parse_args()
        if args.img=='':
            print('请先输入图片')
        else:
            predict(args.img)