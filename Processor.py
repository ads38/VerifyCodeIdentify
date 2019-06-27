# 验证码图片处理模块
# 导入
import CreateVerifyCode as createV
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

############################################################################
# 这里是全局变量定义区域
# 变量（两个）赋值，返回验证码文本信息，和图形对象
text, image = createV.gen_verifycode_pic()

# 图像规格 180 * 67
IMAGE_WIDTH = 180  # 宽度
IMAGE_HEIGHT = 67  # 高度

# 验证码最大文本长度
MAX_CAPTCHA_LEN = len(text)

# 验证码字符集, 数字，大写字母，小写字母，’_ ‘是为了补齐
char_set = createV.number + createV.ALPHABET + createV.alphabet + ['_']

# 验证码字符集总长度
CHAR_SET_LEN = len(char_set)

# 全局变量定义 end
############################################################################

# 打印信息
print("验证码图像 channel ", image.shape)  # (60, 160, 3)
'''
 注：image.shape[0] 图像垂直垂直尺寸
    image.shape[1] 图像水平尺寸
    image.shape[2] 图像通道数 
'''


# 将目标图像转为灰度图像
# 参数1：目标图像对象
def convert2gray(target_img):
    # shape 长度大于 2，彩色图形的维度，长、宽、色彩分量数
    if (len(target_img.shape) > 2):
        # 灰度值
        gray = np.mean(target_img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return target_img


# 文本转换为向量
# 参数1：目标文本字符串
def text2vec(target_txt):
    # 获取文本长度
    text_len = len(target_txt)
    # 如果超过
    if text_len > MAX_CAPTCHA_LEN:
        raise ValueError('长度超过限制！')

    # 向量,其实是矩阵初始化
    vector = np.zeros(MAX_CAPTCHA_LEN * CHAR_SET_LEN)

    # 字符通过ASCII转换为位置（在矩阵中的下标）
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(target_txt):
        index = i * CHAR_SET_LEN + char2pos(c)
        vector[index] = 1
    return vector


# 向量转回文本内容
def vec2text(target_vec):
    char_pos = target_vec.nonzero()[0]
    txt = []
    for i, c in enumerate(char_pos):

        char_index = c % CHAR_SET_LEN
        if char_index < 10:
            char_code = char_index + ord('0')
        elif char_index < 36:
            char_code = char_index - 10 + ord('A')
        elif char_index < 62:
            char_code = char_index - 36 + ord('a')
        elif char_index == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        txt.append(chr(char_code))
    return "".join(txt)


# 生成批处理（批量训练）函数
def gen_train_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA_LEN * CHAR_SET_LEN])

    # 有时候的图像不为（这里应该是67，180，3而不是网上写的60，160，4）
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = createV.gen_verifycode_pic()  # 读取本地打好标记验证码

            # （这里应该是67，180，3而不是网上写的60，160，4），否则会有死循环
            if image.shape == (67, 180, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        # 转换成灰度图像
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


# 占位符
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA_LEN * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([9 * 23 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA_LEN * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA_LEN * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def train_crack_captcha_cnn():
    start_time = time.time()
    output = crack_captcha_cnn()
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    # learning_rate=0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = gen_train_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = gen_train_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(u'***************************************************************第%s次的准确率为%s' % (step, acc))
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.9:  # 我这里设了0.9，设得越大训练要花的时间越长，如果设得过于接近1，很难达到。如果使用cpu，花的时间很长，cpu占用很高电脑发烫。
                    print("保存模型......")
                    # joblib.dump(sess, "F:/model/rf_model.m")
                    saver.save(sess, "D:/model/crack_capcha.model", global_step=step)
                    print(time.time() - start_time)
                    break
            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('.'))
        # saver.restore(sess, "F:/CNN_1/model/crack_capcha.model-10400")
        '''
            D 盘里的模型文件（路径D:/model/），精度在 90.5%
            总共4个文件
            D:/model/checkpoint
            D:/model/crack_capcha.model-30900.data-00000-of-00001
            D:/model/crack_capcha.model-30900.index
            D:/model/crack_capcha.model-30900.meta
            在 以下的saver.restore()函数中， 命名为 crack_capcha.model-30900
        '''
        saver.restore(sess, "D:/model/crack_capcha.model-30900")

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        # text_list = sess.run(predict, feed_dict={X: [captcha_image]})

        text = text_list[0].tolist()
        vector = np.zeros(MAX_CAPTCHA_LEN * CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * CHAR_SET_LEN + n] = 1
            i += 1
        return vec2text(vector)



##################################################
root_dir = "D:\\pics"
img_list = []

def gen_list():
    for parent, dirnames, filenames in os.walk(root_dir): # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        for filename in filenames: # 输出文件信息
            img_list.append(filename.replace(".jpg", ""))
    return img_list

img_list = gen_list()
def get_test_captcha_text_and_image(i=None):
    img = img_list[i]
    captcha_image = Image.open(root_dir + "\\" + img + ".jpg")
    captcha_image = np.array(captcha_image)
    return img, captcha_image


# 向量转字符
def vec2chr(num):
    char_index = num % CHAR_SET_LEN
    if char_index < 10:
        char_code = char_index + ord('0')
    elif char_index < 36:
        char_code = char_index - 10 + ord('A')
    elif char_index < 62:
        char_code = char_index - 36 + ord('a')
    elif char_index == 62:
        char_code = ord('_')
    else:
        raise ValueError('error')
    return chr(char_code)



def crack_captcha2():
 output = crack_captcha_cnn()

 saver = tf.train.Saver()
 with tf.Session() as sess:
  saver.restore(sess, "D:/model/crack_capcha.model-30900")

  predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA_LEN, CHAR_SET_LEN]), 2)
  count = 0
  # 因为测试集共500个...写的很草率
  for i in range(500):
   text, image = get_test_captcha_text_and_image(i)
   image = convert2gray(image)
   captcha_image = image.flatten() / 255
   text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
   predict_text = text_list[0].tolist()
   tmp = ''
   # 向量值转字符
   for i in range(4):
       tmp += vec2chr(predict_text[i])

   if str.upper(text) == str.upper(tmp):
    count += 1
    check_result = "，预测结果正确"
    print("正确: {} 预测: {}".format(text, tmp) + check_result)
   else:
    check_result = "，预测结果不正确"
    print("正确: {} 预测: {}".format(text, tmp) + check_result)


 print("正确率:" + str(count) + "/500")

