import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 50

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    tf.summary.histogram('layer_1', layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    tf.summary.histogram('layer_2', layer_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    tf.summary.histogram('out_layer', out_layer)
    return out_layer


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

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    logits = multilayer_perceptron(X)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/multi-perceptron', sess.graph)
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            summary, _ = sess.run([merged, train_op], feed_dict={X: train_vec, Y: train_label})
            writer.add_summary(summary, epoch)
            # Compute average loss
            #avg_cost += c / divide
            # Display logs per epoch step
            if epoch % display_step == 0:
                _, c = sess.run([train_op, accuracy], feed_dict={X: train_vec, Y: train_label})
                print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(c))
        print("Optimization Finished!")

        # Test model
        pred = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: test_vec, Y: test_label}))
        writer.close()
