import numpy as np
import tensorflow as tf
from image_style_transform.vgg_net.vgg19 import VGG19
from scipy.misc import imread, imresize
import scipy.misc
import matplotlib.pyplot as plt
from functools import reduce

# load VGG19 weight
tf.reset_default_graph()
VGG = VGG19.VGG19('모델 경로' + VGG19.MODEL_FILE_NAME)

# load images
path_cot = '원본 이미지 경로'
path_sty = '스타일 이미지 경로'
path_save = '이미지 저장 경로'
img_cot_real = imread('{}'.format(path_cot), mode='RGB')
img_sty_real = imread('{}'.format(path_sty), mode='RGB')
img_cot = imresize(img_cot_real, (512, 512))
img_sty = imresize(img_sty_real, (512, 512))

# befor Placeholder
content_img0 = np.float32(VGG.preprocess([img_cot]))
style_img0 = np.float32(VGG.preprocess([img_sty]))
initial = np.float32(VGG.preprocess([img_cot]))

# placeholder & variable
content_img = tf.placeholder(tf.float32, shape=content_img0.shape)
style_img = tf.placeholder(tf.float32, shape=style_img0.shape)
initial_img = tf.Variable(initial, dtype=tf.float32)

# load network
content_net = VGG.feed_forward(content_img, scope='content')
style_net = VGG.feed_forward(style_img, scope='style')
initial_net = VGG.feed_forward(initial_img, scope='initial')

# each layer define
# content_layers = ['conv4_2']
# style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']


# style_loss_define
def gram_func(tensor):
    shape = tensor.get_shape()
    new_shape = int(shape[3])
    matrix = tf.reshape(tensor, shape=[-1, new_shape])
    gram_matrix = tf.matmul(tf.transpose(matrix), matrix)
    _, H, W, D = tensor.get_shape()
    N = H.value * W.value
    M = D.value
    size = 1 / (M * N * 2)
    return gram_matrix * size


# style loss define
def style_loss(init_net, style_net):
    style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    style_losses = []
    for layer in style_layers:
        gram_loss = tf.nn.l2_loss(gram_func(init_net[layer]) - gram_func(style_net[layer]))
        style_losses.append(gram_loss * 0.2)
    return reduce(tf.add, style_losses * 100)


# content_loss_define
def content_loss(init_net, content_net):
    content_layers = ['conv4_2']
    content_losses = []
    for layer in content_layers:
        content_losses.append(tf.nn.l2_loss(content_net[layer] - init_net[layer]) / 2.)
    return reduce(tf.add, content_losses * 75)


# total_loss
total_loss = content_loss(initial_net, content_net) + style_loss(initial_net, style_net)
total_optimizer = tf.train.AdamOptimizer(learning_rate=30).minimize(total_loss)

# session open & initialize
sess = tf.Session()
init_g = tf.global_variables_initializer()
sess.run(init_g)

# train image style
for i in range(5001):
    _, total = sess.run([total_optimizer, total_loss], feed_dict={content_img: [img_cot], style_img: [img_sty]})
    print('{}_total : {}'.format(i, total))

    if i % 100 == 0:
        image1 = sess.run(initial_img)[0]
        final_image1 = np.clip(VGG.undo_preprocess(image1), 0.0, 255.0)
        # PIL.Image.fromarray(final_image.astype(np.uint8) ).save('C:/Users/Administrator/Desktop/image style/result/' + str(i) + '.jpg', 'jpeg')
        plt.imsave(path_save + '/style_' + str(i) + '.jpg', final_image1.astype(np.uint8))
