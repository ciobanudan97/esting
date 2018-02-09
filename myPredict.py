import tensorflow as tf
import sys
from PIL import Image, ImageFilter



class NeuralNet:
    def __init__(self, nClasses, image_size):
        self.n_classes = nClasses
        self.image_size = image_size
        self.X = tf.placeholder(dtype='float', shape=[None, self.image_size], name='X')  # height, width
        self.Y = tf.placeholder(dtype='int32', name='Y')

    def model(self, num_nodehl1, num_nodehl2, num_nodehl3):
        self.n_nodes_hl1 = num_nodehl1
        self.n_nodes_hl2 = num_nodehl2
        self.n_nodes_hl3 = num_nodehl3

        self.hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([self.image_size, self.n_nodes_hl1])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl1]))}

        self.hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl1, self.n_nodes_hl2])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl2]))}

        self.hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl2, self.n_nodes_hl3])),
                               'biases': tf.Variable(tf.random_normal([self.n_nodes_hl3]))}

        self.output_layer = {'weights': tf.Variable(tf.random_normal([self.n_nodes_hl3, self.n_classes])),
                             'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        # X * weights + biases
        self.l1 = tf.add(tf.matmul(self.X, self.hidden_layer_1['weights']), self.hidden_layer_1['biases'])
        self.l1 = tf.nn.relu(self.l1)

        self.l2 = tf.add(tf.matmul(self.l1, self.hidden_layer_2['weights']), self.hidden_layer_2['biases'])
        self.l2 = tf.nn.relu(self.l2)

        self.l3 = tf.add(tf.matmul(self.l2, self.hidden_layer_3['weights']), self.hidden_layer_3['biases'])
        self.l3 = tf.nn.relu(self.l3)

        self.predicted_class = tf.add(tf.matmul(self.l3, self.output_layer['weights']), self.output_layer['biases'])

        return self.predicted_class


n_classes = 10
image_size = 28*28

def predict(img):
    neuralNet = NeuralNet(n_classes,image_size)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess,"/tmp/model_mnist.ckpt")
        prediction = tf.argmax(neuralNet.model(), 1)
        return prediction.eval(feed_dict={neuralNet.X: [img]}, session=sess)

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheigth = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # caculate horizontal pozition
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png")

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva
    # print(tva)


def main():
    """
    Main function.
    """
    imvalue = imageprepare("C:/Users/dan.ciobanu/Desktop/doi.png")
    predint = predict(imvalue)
    print(predint[0])  # first value in list


if __name__ == "__main__":
    main()