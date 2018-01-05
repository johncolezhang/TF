import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":
    IRIS_TRAINING = "iris_data/iris_training.csv"
    IRIS_TEST = "iris_data/iris_test.csv"

    # Load datasets.
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data, number of feature is 4
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

    # Build 4 layer DNN with 20, 40, 40, 20 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[20, 40, 40, 20],
                                            n_classes=3,
                                            dropout=0.5,
                                            activation_fn=tf.nn.relu,
                                            model_dir="/tmp/iris_model")

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model, training 2000 steps.
    classifier.train(input_fn=train_input_fn, steps=40000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: ", accuracy_score)

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]],
        dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    for p in predictions:
        print(p)

