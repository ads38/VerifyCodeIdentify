# coding=utf-8
# captcha
# numpy
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import random
# import os, glob

# 验证码中包含数字和英文字幕大小写
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z'
           ]

ALPHABET = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z'
           ]

# 验证码图片保存路径
PIC_DIR = 'D:\\pics\\'
# 图片扩展名
EXT = '.jpg'


# 生成纯文本，总计4个字符
# 随机生成的可能有 8Gxb 等等
# 参数1：指定字符集
# 参数2：默认验证码文本长度=4
def gen_verifycode_txt(char_set=number + alphabet + ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成每个字符对应的验证码图片
def gen_verifycode_pic():
    # 指定大小 180 * 67
    img = ImageCaptcha(180, 67)
    # 指定文本，长度4位
    txt = gen_verifycode_txt()
    txt = ''.join(txt)
    # 测试，打印出验证码文本
    # print('-------------- ' + txt)
    # 根据文本生成图像
    captcha = img.generate(txt)
    # 写入文件
    img.write(txt, PIC_DIR + txt + EXT)
    captcha_image = Image.open(captcha)

    # 图片显示出来
    # captcha_image.show()
    captcha_image = np.array(captcha_image)

    # 返回最终图形对象和文字
    return txt, captcha_image



