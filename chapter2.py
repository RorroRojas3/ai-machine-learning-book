import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential({
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten takes square value of 2D array to 1D array,
    # relu returns value if greater than 0
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 128 is arbitrary, no fixed number of neurons to use
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 neurons because we have 10 classes, softmax picks the
    # highest value
})

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
