import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform

def load_img(path):
    img = skimage.io.imread(path)

    img = img.reshape(224,224,1)
    img = np.concatenate((img, img, img), -1)

    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224), mode='constant')[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def load_data():
    imgs = {'level3': [], 'level2': []}
    for k in imgs.keys():
        dir = './data/' + k
        for file in os.listdir(dir):
            if not file.lower().endswith('.png'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            imgs[k].append(resized_img)    

    level3_y = np.tile([200],(len(imgs['level3']),1))
    level2_y = np.tile([0],(len(imgs['level2']),1))
    return imgs['level3'], imgs['level2'], level3_y, level2_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16/VGG19 parameters')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    def predict(self, paths):
        correct= 0
        threshold = 90
        for i, path in enumerate(paths):
            x = load_img(path)
            length = self.sess.run(self.out, {self.tfx: x})
            print(length,path)
            if 'C1' in path and length< threshold:
                correct = correct + 1
            elif 'C2' in path and length> threshold:
                correct = correct + 1
            else:
                print('Error!')
        print('correct : ', correct)

    def save(self, path='./model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    level3_x, level2_x, level3_y, level2_y = load_data()
 
    xs = np.concatenate(level3_x + level2_x, axis=0)
    ys = np.concatenate((level3_y, level2_y), axis=0)

    vgg = Vgg16(vgg16_npy_path='./vgg16.npy')
    print('Net built')
    for i in range(1000):
        b_idx = np.random.randint(0, len(xs), 100)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)
	if train_loss < 400:
	    break
    vgg.save('./model/transfer_learn')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='./vgg16.npy',
                restore_from='./model/transfer_learn')
    vgg.predict(
        ['./data/level2/C1_078.png', './data/level3/C2_087.png'])

def test():
    num_input_test=0
    files_t=[]
    for filename_t in os.listdir('../CBIS-DDSM/Two_Hist2_CC_cut_3000_X3_test'):
        if num_input_test >=4000:
            break
        num_input_test += 1
        files_t.append(filename_t)
    vgg = Vgg16(vgg16_npy_path='./vgg16.npy',
                restore_from='./model/transfer_learn')
    vgg.predict(files_t)

if __name__ == '__main__':
    train()
    #eval()
    #test()
