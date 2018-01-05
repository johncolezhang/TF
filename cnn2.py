import tensorflow as tf
import numpy as np



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #strides = [1, stride, stride, 1], step size.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def generate_label(label):
    length = len(label)
    label_mat = np.zeros((length, 10))
    for i in range(length):
        label_mat[i, int(label[i])] = 1
    return label_mat

def next_batch(vec, label, num):
    idx = np.arange(0, len(vec))
    np.random.shuffle(idx)
    idx = idx[:num]
    first = True
    for i in idx:
        if first:
            vec_shuffle = vec[i, :]
            label_shuffle = label[i, :]
            first = False
        else:
            vec_shuffle = np.vstack((vec_shuffle, vec[i, :]))
            label_shuffle = np.vstack((label_shuffle, label[i, :]))
    return vec_shuffle, label_shuffle


if __name__ == "__main__":
    #mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    divide = 2000
    vec = np.matrix(np.loadtxt("digits/digits4000_digits_vec.txt"))
    label = np.loadtxt("digits/digits4000_digits_labels.txt")
    train_vec = vec[0: divide, :]
    train_label = generate_label(label[0: divide])
    test_vec = vec[divide: 2 * divide, :]
    test_label = generate_label(label[divide: 2 * divide])

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    #first layer
    #pre activate
    #compute 32 features for each 5x5 patch
    #[filter_height, filter_width, in_channels, out_channels]
    #output 32 features
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    #reshape
    #-1 = x.row * x.column / (28 * 28 * 1)
    #in this case, -1 means the number of digits
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #activate using relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    #pooling
    #The max_pool_2x2 method will reduce the image size to 14 * 14.
    h_pool1 = max_pool_2x2(h_conv1)

    #second layer
    #output 64 features
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    #reduce image size to 7 * 7
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #densely fully connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #output layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            #randomly choose 50 digits to train.
            batch_vec, batch_label = next_batch(train_vec, train_label, 50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: train_vec, y_: train_label, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            #dropout rate is 0.5.
            train_step.run(feed_dict={x: batch_vec, y_: batch_label, keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_vec, y_: test_label, keep_prob: 1.0}))

