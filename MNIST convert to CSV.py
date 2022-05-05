# 将pytorch中mnist数据集的图像可视化及保存
# https://www.jb51.net/article/178405.htm
import torch
import torchvision
import torch.utils.data as Data
import scipy.misc
import os
import matplotlib.pyplot as plt
#将MNIST数据集转换为csv格式

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


convert("./dataset/MNIST/raw/train-images-idx3-ubyte", "./dataset/MNIST/raw/train-labels-idx1-ubyte",
        "./dataset/mnist_train.csv", 60000)
convert("./dataset/MNIST/raw/t10k-images-idx3-ubyte", "./dataset/MNIST/raw/t10k-labels-idx1-ubyte",
        "./dataset/mnist_test.csv", 10000)

print("Convert Finished!")
