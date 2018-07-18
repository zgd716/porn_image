import os
from image_utils import *
import tensorflow as tf
import numpy as np

class Preprocess_Image():

    def __init__(self,sess,porn_path,unporn_path,batch_size):
        self.sess=sess
        self.porn_path=porn_path
        self.unporn_path=unporn_path
        self.batch_size=batch_size

    def build_input(self):
        self.images=[]
        self.labels=[]
        fn_load_image = create_tensorflow_image_loader(self.sess)
        porn_list=os.listdir(self.porn_path)
        for porn_img in porn_list:
            self.images.append(tf.squeeze(fn_load_image(os.path.join(self.porn_path,porn_img))).eval(session=self.sess))
            self.labels.append([0,1])
        unporn_list=os.listdir(self.unporn_path)
        for unporn_img in unporn_list:
            self.images.append(tf.squeeze(fn_load_image(os.path.join(self.unporn_path,unporn_img))).eval(session=self.sess))
            self.labels.append([1,0])

        self.images=np.array(self.images)
        self.labels=np.array(self.labels)

    def get_batch(self):

        self.build_input()
        batches=len(self.labels)//self.batch_size
        self.y=self.labels[:batches*self.batch_size]
        self.x= self.images[:batches * self.batch_size]
        print(self.y.shape)
        indexs=np.random.permutation(len(self.y))
        self.x=self.x[indexs]
        self.y=self.y[indexs]
        print(self.y.shape)
        for i in range(batches):
             yield self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size]