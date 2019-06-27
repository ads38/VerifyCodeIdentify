import CreateVerifyCode as cvc
import Processor as pcs
import os
from captcha.image import ImageCaptcha # pip install captcha
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import os
from random import choice

# 屏蔽系统警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 500张（可选）
#for i in range(500):
 #   cvc.gen_verifycode_pic()

# 训练函数，跑了10个小时
# pcs.train_crack_captcha_cnn()


pcs.crack_captcha2()

