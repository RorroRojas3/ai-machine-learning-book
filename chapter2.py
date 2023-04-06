import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback) :
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callBacks = myCallback()

data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Flatten takes square value of 2D array to 1D array, relu returns value if greater than 0
# 128 is arbitrary, no fixed number of neurons to use
# 10 neurons because we have 10 classes, softmax picks the highest value

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callBacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])