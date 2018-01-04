import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    divide = 2000
    vec = np.matrix(np.loadtxt("digits/digits4000_digits_vec.txt"))
    label = np.loadtxt("digits/digits4000_digits_labels.txt")
    train_vec = vec[0: divide, :]
    train_label = label[0: divide]
    test_vec = vec[divide: 2 * divide, :]
    test_label = label[divide: 2 * divide]


    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])
    # Nearest Neighbor calculation using L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), 1)
    pred = tf.argmin(distance, 0)
    accuracy = 0
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(test_label)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: train_vec, xte: test_vec[i].A1})
            print("Test ", i, " Prediction:", train_label[nn_index], "True Class:", test_label[i])
            if train_label[nn_index] == test_label[i]:
                accuracy += 1
        print("accuracy: ", accuracy / len(train_label))
