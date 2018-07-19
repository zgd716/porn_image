# porn_image

该项目参考yahoo开源的nsfw项目；在预训练基础上进行的fine tuning。

两种使用方式：

1、直接运行 python classify_nsfw.py -m data/open_nsfw-weights.npy xxx.jpg就可以进行图片预测；


2、在data文件夹下的porn和unporn中分别存放训练的样本；执行python  train.py；模型即可生成；使用python predict.py -i xx.jpg即可进行预测

note:

1、方式1使用的是nsfw的预训练参数

2、方式2使用的是nsfw的fine tuning
