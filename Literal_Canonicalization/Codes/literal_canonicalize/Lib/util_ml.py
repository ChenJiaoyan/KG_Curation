# This file defines functions for ML training and prediction
import os
import datetime
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


# Train an AttBiRNN
def rnn_train(x_train, y_train, PARAMETERS, rnn_dir):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if PARAMETERS.nn_type == 'AttBiRNN':
                rnn = AttBiRNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                               channel_num=x_train.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size,
                               attention_size=PARAMETERS.attention_size)
            elif PARAMETERS.nn_type == 'BiRNN':
                rnn = BiRNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                            channel_num=x_train.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size)
            else:
                rnn = MLP(sequence_length=x_train.shape[1], num_classes=y_train.shape[1], channel_num=x_train.shape[2])

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(rnn_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(rnn_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_x_batch, train_y_batch):
                feed_dict = {
                    rnn.input_x: train_x_batch,
                    rnn.input_y: train_y_batch,
                    rnn.dropout_keep_prob: PARAMETERS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % PARAMETERS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def train_evaluate(train_x_all, train_y_all):
                feed_dict = {
                    rnn.input_x: train_x_all,
                    rnn.input_y: train_y_all,
                    rnn.dropout_keep_prob: PARAMETERS.dropout_keep_prob
                }
                loss, accuracy = sess.run([rnn.loss, rnn.accuracy], feed_dict)
                print("\t train loss {:g}, train acc {:g}".format(loss, accuracy))

            batches = batch_iter(list(zip(x_train, y_train)), PARAMETERS.num_epochs, PARAMETERS.batch_size)
            current_step = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

            train_evaluate(x_train, y_train)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\t Saved model checkpoint to {}\n".format(os.path.basename(path)))


# Predict with a trained AttBiRNN
# input: test_x (each row represents one sample vector)
#        need_finetune (True or False
#        x_finetune, y_finetune (samples for fine tuning)
# output: test_p (an array, each item represents the score of one sample)
def rnn_predict(test_x, rnn_dir, need_ft=False, x_ft=None, y_ft=None, batch_size=0, num_epochs=0,
                dropout_keep_prob=0):
    checkpoint_dir = os.path.join(rnn_dir, 'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            n_input_x = graph.get_operation_by_name("input_x").outputs[0]
            n_input_y = graph.get_operation_by_name("input_y").outputs[0]
            n_dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            n_scores = graph.get_operation_by_name("output/scores").outputs[0]
            n_alphas = graph.get_operation_by_name("Attention_layer/alphas").outputs[0]

            # fine tune
            if need_ft:
                n_loss = graph.get_operation_by_name("loss/loss").outputs[0]
                n_accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
                train_op = graph.get_operation_by_name("train_op").outputs[0]
                batches = batch_iter(list(zip(x_ft, y_ft)), num_epochs, batch_size)
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    sess.run([train_op], {n_input_x: x_batch, n_input_y: y_batch,
                                          n_dropout_keep_prob: dropout_keep_prob})
                loss, accuracy = sess.run([n_loss, n_accuracy], {n_input_x: x_ft, n_input_y: y_ft,
                                                                 n_dropout_keep_prob: dropout_keep_prob})
                print("\t fine tuning data, train loss {:g}, train acc {:g}".format(loss, accuracy))

            # predict
            # test_y = sess.run(n_scores, {n_input_x: test_x, n_dropout_keep_prob: 1.0})
            test_y, alphas = sess.run([n_scores, n_alphas], {n_input_x: test_x, n_dropout_keep_prob: 1.0})

    return test_y, alphas


# Generate batches of the samples
# In each epoch, samples are traversed one time batch by batch
def batch_iter(data, num_epochs, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            batch_shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[batch_shuffle_indices]
        else:
            shuffled_data = data

        if num_batches > 0:
            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
        else:
            yield shuffled_data


# AttBiRNN
class AttBiRNN(object):

    @staticmethod
    def attention(inputs, attention_size, time_major=False, return_alphas=True):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name="w_omega")
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="b_omega")
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="u_omega")
        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        output = inputs * tf.expand_dims(alphas, -1)
        output = tf.reshape(output, [-1, output.shape[1] * output.shape[2]])
        return output if not return_alphas else output, alphas

    def __init__(self, sequence_length, num_classes, channel_num, rnn_hidden_size, attention_size):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Bidirectional RNN
        self.rnn_outputs, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                     inputs=self.input_x, dtype=tf.float32)

        # Attention layer and a dropout layer
        with tf.name_scope('Attention_layer'):
            self.att_output, alphas = self.attention(inputs=self.rnn_outputs, attention_size=attention_size)
            tf.summary.histogram('alphas', alphas)
        with tf.name_scope("dropout"):
            self.att_drop = tf.nn.dropout(self.att_output, self.dropout_keep_prob, name="dropout")

        # FC layer
        with tf.name_scope("output"):
            #            FC_W = tf.get_variable("FC_W", shape=[rnn_hidden_size * 2, num_classes],
            #                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_W = tf.get_variable("FC_W", shape=[sequence_length * rnn_hidden_size * 2, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.nn.xw_plus_b(self.att_drop, FC_W, FC_b, name="FC_out")
            self.scores = tf.nn.softmax(self.fc_out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name='loss')

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# BiRNN
class BiRNN(object):

    def __init__(self, sequence_length, num_classes, channel_num, rnn_hidden_size):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Bidirectional RNN
        self.rnn_outputs, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                     inputs=self.input_x, dtype=tf.float32)
        self.rnn_output = tf.concat(self.rnn_outputs, 2)
        self.rnn_output_mean = tf.reduce_mean(self.rnn_output, axis=1)

        # FC layer
        with tf.name_scope("output"):
            FC_W = tf.get_variable("FC_W", shape=[rnn_hidden_size * 2, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.nn.xw_plus_b(self.rnn_output_mean, FC_W, FC_b, name="FC_out")
            self.scores = tf.nn.softmax(self.fc_out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name='loss')

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# MLP
class MLP(object):

    def __init__(self, sequence_length, num_classes, channel_num):
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # FC layer
        with tf.name_scope("output"):
            FC_W = tf.get_variable("FC_W", shape=[sequence_length * channel_num, num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            FC_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.nn.xw_plus_b(tf.reshape(self.input_x, [-1, sequence_length * channel_num]),
                                          FC_W, FC_b, name="FC_out")
            self.scores = tf.nn.softmax(self.fc_out, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name='loss')

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
