import numpy as np
from collections import deque

def frange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step

xs = np.array([x for x in frange(0, 50, 0.003)])

ys = np.sin(xs)

def preprocess_data(data):
    xs_q = deque(maxlen=24)
    ys_q = deque(maxlen=1)

    train_x = []
    train_y = []
    val_x = []
    val_y = []

    train_perc = 0.8
    for i in range(len(xs) - 24):
        xs_q.append([])
        ys_q.append([])

        xs_q[len(xs_q) - 1].append(ys[i])
        ys_q[len(ys_q) - 1].append(ys[i + 1])

        if len(xs_q) == 24:
            if i < len(xs) * train_perc:
                # Training data
                train_x.append(np.array(xs_q))
                train_y.append(np.array(ys_q))
            else:
                # Testing data
                val_x.append(np.array(xs_q))
                val_y.append(np.array(ys_q))

    return ((np.array(train_x), np.array(train_y)), 
    (np.array(val_x), np.array(val_y)))

(train_x, train_y), (val_x, val_y) = preprocess_data(ys)
print("train_x.shape:", train_x.shape)
print("train_y.shape:", train_y.shape)
print("val_x.shape:", val_x.shape)
print("val_y.shape:", val_y.shape)

rng_state = np.random.get_state()
np.random.shuffle(train_x)
np.random.set_state(rng_state)
np.random.shuffle(train_y)

import tensorflow as tf

EPOCHS = 5
OUTPUT_DIM = 1
INPUT_DIM = 1
BACTH_SIZE = 64
RNN_SIZE = 128
TIME_STEPS = 24

x = tf.placeholder(tf.float32, [None, TIME_STEPS, INPUT_DIM])
y = tf.placeholder(tf.float32)

def rnn(x):
    weights = tf.Variable(tf.random_normal([RNN_SIZE, 1 * OUTPUT_DIM]))
    biases = tf.Variable(tf.random_normal([1 * OUTPUT_DIM]))

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 1])
    x = tf.split(x, TIME_STEPS, 0)
    
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_SIZE, state_is_tuple=True)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.nn.tanh(tf.matmul(tf.nn.tanh(outputs[-1]), weights + biases))

    return tf.reshape(output, [-1, 1, 1])

prediction = rnn(x)
cost = tf.reduce_mean(tf.abs(prediction - y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

import matplotlib.pyplot as plt
import time

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCHS):
        epoch_time = time.clock_gettime(0);
        # epoch_loss = 0
        for i in range(len(train_x) // BACTH_SIZE):
            epoch_x = train_x[i * BACTH_SIZE:(i + 1) * BACTH_SIZE]
            epoch_y = train_y[i * BACTH_SIZE:(i + 1) * BACTH_SIZE]
            
            sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            # _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            # epoch_loss += c

        c = sess.run(cost, feed_dict={x: val_x, y: val_y})
        print('Epoch', epoch + 1, 'completed out of', EPOCHS, 'loss:', c)
        print("Time spended for epoch:", time.clock_gettime(0) - epoch_time)

    predictions = []
    real = []

    import matplotlib.pyplot as plt

    def append_to_end(xs, prediction):
        for i in range(len(xs) - 1):
            xs[i] = xs[i + 1]
        xs[-1] = np.array(prediction)

    for i in range(len(val_x) // TIME_STEPS):
        xs = val_x[i * TIME_STEPS:i * TIME_STEPS + 1]
        for j in range(TIME_STEPS):
            prediction_ = np.reshape(sess.run(prediction, feed_dict={x: xs}), [-1])
            predictions.extend(prediction_)
            real.extend(np.reshape(val_y[i * TIME_STEPS + j], [-1]))
            append_to_end(xs[0], prediction_)

    plt.plot(predictions, label='predictions')
    plt.plot(real, label='real')
    plt.legend(loc='upper left')
    plt.show()
