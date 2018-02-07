import tensorflow as tf
import abc


class BaseVAE(object):
    def __init__(self, hparams, x, dropout):

        self.x = x
        self.dropout = dropout
        self.targets = x
        self.x_dim = hparams.x_dim
        self.z_dim = hparams.z_dim
        self.output_dim = self.x_dim
        self.hidden_dim = hparams.hidden_dim
        # Initializer
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.constant_initializer(0.)

        self.z, self.outputs, self.loss = self.build_graph(hparams)
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d,"
              "decay_factor %g" % (hparams.start_decay_step, hparams.learning_rate,
                                   hparams.decay_steps, hparams.decay_factor))
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        if hparams.optimizer == "sgd":
            # perform SGD with a learning rate with exponential decay
            self.learning_rate = tf.cond(
                self.global_step < hparams.start_decay_step,
                lambda: tf.constant(hparams.learning_rate),
                lambda: tf.train.exponential_decay(
                    hparams.learning_rate,
                    (self.global_step - hparams.start_decay_step),
                    hparams.decay_steps,
                    hparams.decay_factor,
                    staircase=True),
                name="learning_rate")
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            tf.summary.scalar("lr", self.learning_rate)
        elif hparams.optimizer == "adam":
            self.learning_rate = tf.constant(hparams.learning_rate)
            opt = tf.train.AdamOptimizer(self.learning_rate)

        self.update = opt.minimize(self.loss, global_step=self.global_step)
        # Summary
        self.train_summary = tf.summary.merge([
            tf.summary.scalar("lr", self.learning_rate),
            tf.summary.scalar("train_loss", self.loss),
        ])
        # Saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        print("# Trainable variables")
        for param in params:
            print("  %s, %s" % (param.name, str(param.get_shape())))

    def build_graph(self, hparams):

        with tf.variable_scope('vae'):
            mean, sigma = self._build_encoder(hparams)
            z = self._sampling(mean, sigma)
            outputs = self._build_decoder(hparams, z)
            # outputs = tf.clip_by_value(outputs, 1e-8, 1 - 1e-8)
            loss = self._compute_loss(outputs, self.targets, mean, sigma)
        return z, outputs, loss

    def train(self, sess):
        return sess.run([self.update,
                         self.loss,
                         self.train_summary,
                         self.global_step,
                         self.x,
                         self.targets,
                         self.outputs])

    def eval(self, sess):
        return sess.run(self.loss)

    def infer(self, sess):
        return sess.run([self.outputs, self.z, self.x])

    @abc.abstractmethod
    def _compute_loss(self, outputs, targets, mean, sigma, epsilon=1e-8):
        pass

    @abc.abstractmethod
    def _build_encoder(self, hparams):
        pass

    def _sampling(self, mean, sigma):
        with tf.variable_scope('sampling'):
            e = tf.random_normal(tf.shape(mean), mean=0, stddev=1, dtype=tf.float32)
        return tf.add(mean, tf.multiply(sigma, e))

    @abc.abstractmethod
    def _build_decoder(self, hparams, z):
        pass


class FullyConnectedVAE(BaseVAE):
    def _build_encoder(self, hparams):
        with tf.variable_scope('gaussian_mlp_encoder'):
            f1 = tf.contrib.layers.fully_connected(self.x, hparams.hidden_dim, activation_fn=tf.nn.elu,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.bias_initializer)
            f1 = tf.layers.dropout(f1,rate=self.dropout)
            f2 = tf.contrib.layers.fully_connected(f1, hparams.hidden_dim, activation_fn=tf.nn.tanh,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.bias_initializer)
            f2 = tf.layers.dropout(f2,rate=self.dropout)
            output = tf.contrib.layers.fully_connected(f2, self.z_dim * 2, activation_fn=None,
                                                       weights_initializer=self.weights_initializer,
                                                       biases_initializer=self.bias_initializer)
            mean = output[:, :self.z_dim]
            sigma = 1e-6 + tf.nn.softplus(output[:, self.z_dim:])
        return mean, sigma

    def _build_decoder(self, hparams, z):
        with tf.variable_scope('bernoulli_mlp_decoder'):
            f1 = tf.contrib.layers.fully_connected(z, hparams.hidden_dim, activation_fn=tf.nn.tanh,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.bias_initializer)
            f1 = tf.layers.dropout(f1,rate=self.dropout)
            f2 = tf.contrib.layers.fully_connected(f1, hparams.hidden_dim, activation_fn=tf.nn.elu,
                                                   weights_initializer=self.weights_initializer,
                                                   biases_initializer=self.bias_initializer)
            f2 = tf.layers.dropout(f2,rate=self.dropout)
            outputs = tf.contrib.layers.fully_connected(f2, self.output_dim, activation_fn=tf.nn.sigmoid,
                                                        weights_initializer=self.weights_initializer,
                                                        biases_initializer=self.bias_initializer)
        return outputs

    def _compute_loss(self, outputs, targets, mean, sigma, epsilon=1e-10):
        with tf.variable_scope('loss'):
            # log_loss = tf.losses.log_loss(targets,outputs)
            log_loss = -tf.reduce_sum(
                targets * tf.log(outputs + epsilon) + (1 - targets) * tf.log(1 - outputs + epsilon), axis=1)
            var = tf.square(sigma)
            kl_divergence = -0.5 * tf.reduce_sum(1 + tf.log(var + epsilon) - tf.square(mean) - var, axis=1)

        return tf.reduce_mean(kl_divergence + log_loss)
