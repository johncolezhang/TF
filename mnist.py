import tensorflow as tf
import numpy as np

step = 1000
display_step = 50

def generate_label(label):
    length = len(label)
    label_mat = np.zeros((length, 10))
    for i in range(length):
        label_mat[i, int(label[i])] = 1
    return label_mat



if __name__ == "__main__":
    divide = 2000
    vec = np.matrix(np.loadtxt("digits/digits4000_digits_vec.txt"))
    label = np.loadtxt("digits/digits4000_digits_labels.txt")
    train_vec = vec[0: divide, :]
    train_label = generate_label(label[0: divide])
    test_vec = vec[divide: 2 * divide, :]
    test_label = generate_label(label[divide: 2 * divide])


    ##################   softmax regression  ########################
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b
    tf.summary.histogram('y', y)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/mnist', sess.graph)
    for i in range(step):
        summary, _ = sess.run([merged, train_step], feed_dict={x: train_vec, y_: train_label})
        writer.add_summary(summary, i)
        if i % display_step == 0:
            _, c = sess.run([train_step, accuracy], feed_dict={x: train_vec, y_: train_label})
            print("current step: ", i, ", current training accuracy: ", c)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: test_vec, y_: test_label}))
    writer.close()
