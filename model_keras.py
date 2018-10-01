import random

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input, CuDNNLSTM, LSTM, Flatten, Concatenate, concatenate
from keras import backend as K
from keras.regularizers import l2

class MixtureModel():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        def get_cell():
            return Graph.CuDNNLSTM(args.rnn_size, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
                bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None,
                bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None,
                bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

        def dropout():
            return Dropout(args.keep_prob)

        #input
        self.input = Input(name='input', shape=(args.seq_length,3), dtype='float32')

        # two LSTM layers
        self.lstm1 = LSTM(args.rnn_size,return_sequences=True)(self.input)
        self.dropout1 = Dropout(args.keep_prob)(self.lstm1)
        self.lstm2 = LSTM(args.rnn_size)(self.dropout1)
        self.dropout2 = Dropout(args.keep_prob)(self.lstm2)

        self.num_mixture = args.num_mixture
        # end_of_stroke + prob + 2*(mu + sig) + corr
        NOUT = 1 + self.num_mixture * 6
        self.linear_layer = Dense(NOUT, activation="linear")(self.dropout2)

        self.mux = Dense(self.num_mixture)(self.linear_layer)
        self.muy = Dense(self.num_mixture)(self.linear_layer)

        # exponentiate the sigmas and also make corr between -1 and 1.
        #z_sigma1 = tf.exp(z_sigma1)
        #z_sigma2 = tf.exp(z_sigma2)
        self.sigmax = Dense(self.num_mixture,activation=K.exp, kernel_regularizer=l2(1e-3))(self.linear_layer)
        self.sigmay = Dense(self.num_mixture, activation=K.exp, kernel_regularizer=l2(1e-3))(self.linear_layer)
        #z_corr = tf.tanh(z_corr)
        self.rho = Dense(self.num_mixture, activation='tanh')(self.linear_layer)
        """softmax all the pi's:
        max_pi = tf.reduce_max(z_pi, 1, keepdims=True)
        z_pi = tf.subtract(z_pi, max_pi)
        z_pi = tf.exp(z_pi)
        normalize_pi = tf.reciprocal(
            tf.reduce_sum(z_pi, 1, keep8dims=True))
        z_pi = tf.multiply(normalize_pi, z_pi)
        """
        self.pi = Dense(self.num_mixture, activation='softmax')(self.linear_layer)

        # end of stroke signal
        #z_eos = tf.sigmoid(z_eos)  # should be negated, but doesn't matter.
        self.eos = Dense(1, activation='sigmoid')(self.linear_layer)
        # We can then concatenate the two vectors:
        self.outputs = concatenate([self.eos,self.pi,self.mux,self.muy, self.sigmax,self.sigmay,self.rho], axis=-1, name = "Output")

        self.model = Model(inputs=[self.input], outputs=[self.outputs])

        # ----- build mixture density cap on top of second recurrent cell
        def gaussian2d(x1, x2, mu1, mu2, s1, s2, rho):
            # define gaussian mdn (eq 24, 25 from http://arxiv.org/abs/1308.0850)
            x_mu1 = tf.subtract(x1, mu1)
            x_mu2 = tf.subtract(x2, mu2)
            Z = tf.square(tf.div(x_mu1, s1)) + \
                tf.square(tf.div(x_mu2, s2)) - \
                2 * tf.div(tf.multiply(rho, tf.multiply(x_mu1, x_mu2)), tf.multiply(s1, s2))
            rho_square_term = 1 - tf.square(rho)
            power_e = tf.exp(tf.div(-Z, 2 * rho_square_term))
            regularize_term = 2 * np.pi * tf.multiply(tf.multiply(s1, s2), tf.sqrt(rho_square_term))
            gaussian = tf.div(power_e, regularize_term)
            return gaussian

        def gaussian_loss(y_true, y_pred):
            # define loss function (eq 26 of http://arxiv.org/abs/1308.0850)
            x_data , y_data = y_true[:,1:]
            eos_data = y_true[:,0:1]
            pi,mux, muy, sigmax, sigmay, rho = K.split(y_pred[:,1:],6, 1)
            eos = y_pred[:,0:1]
            gaussian = gaussian2d(x_data , y_data , mux, muy, sigmax, sigmay, rho)
            term1 = K.multiply(gaussian, pi)
            term1 = K.reduce_sum(term1, 1, keep_dims=True)  # do inner summation
            term1 = -K.log(K.maximum(term1, 1e-20))  # some errors are zero -> numerical errors.

            term2 = K.multiply(eos, eos_data) + K.multiply(1 - eos,
                                                             1 - eos_data)  # modified Bernoulli -> eos probability
            term2 = -K.log(term2)  # negative log error gives loss

            return K.reduce_sum(term1 + term2)  # do outer summation

        self.model.compile(optimizer='rmsprop', loss={'Output': gaussian_loss})

        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max



        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
                2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
            negRho = 1 - tf.square(rho)
            result = tf.exp(tf.div(-z, 2 * negRho))
            denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(negRho))
            result = tf.div(result, denom)
            return result

        def get_lossfunc(
                z_pi,
                z_mu1,
                z_mu2,
                z_sigma1,
                z_sigma2,
                z_corr,
                z_eos,
                x1_data,
                x2_data,
                eos_data):
            result0 = tf_2d_normal(
                x1_data,
                x2_data,
                z_mu1,
                z_mu2,
                z_sigma1,
                z_sigma2,
                z_corr)
            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            epsilon = 1e-20
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keepdims=True)
            # at the beginning, some errors are exactly zero.
            result1 = -tf.log(tf.maximum(result1, 1e-20))

            result2 = tf.multiply(z_eos, eos_data) + \
                tf.multiply(1 - z_eos, 1 - eos_data)
            result2 = -tf.log(result2)

            result = result1 + result2
            return tf.reduce_sum(result)

        # below is where we need to do MDN splitting of distribution params
        def get_mixture_coef(output):
            # returns the tf slices containing mdn dist params
            # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
            z = output
            z_eos = z[:, 0:1]
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(
                axis=1, num_or_size_splits=6, value=z[:, 1:])

            # process output z's into MDN paramters

            # end of stroke signal
            z_eos = tf.sigmoid(z_eos)  # should be negated, but doesn't matter.

            # softmax all the pi's:
            max_pi = tf.reduce_max(z_pi, 1, keepdims=True)
            z_pi = tf.subtract(z_pi, max_pi)
            z_pi = tf.exp(z_pi)
            normalize_pi = tf.reciprocal(
                tf.reduce_sum(z_pi, 1, keep8dims=True))
            z_pi = tf.multiply(normalize_pi, z_pi)

            # exponentiate the sigmas and also make corr between -1 and 1.
            z_sigma1 = tf.exp(z_sigma1)
            z_sigma2 = tf.exp(z_sigma2)
            z_corr = tf.tanh(z_corr)

            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos]

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2,
            o_corr, o_eos] = get_mixture_coef(output)

        # I could put all of these in a single tensor for reading out, but this
        # is more human readable
        data_out_pi = tf.identity(o_pi, "data_out_pi")
        data_out_mu1 = tf.identity(o_mu1, "data_out_mu1")
        data_out_mu2 = tf.identity(o_mu2, "data_out_mu2")
        data_out_sigma1 = tf.identity(o_sigma1, "data_out_sigma1")
        data_out_sigma2 = tf.identity(o_sigma2, "data_out_sigma2")
        data_out_corr = tf.identity(o_corr, "data_out_corr")
        data_out_eos = tf.identity(o_eos, "data_out_eos")

        # sticking them all (except eos) in one op anyway, makes it easier for freezing the graph later
        # IMPORTANT, this needs to stack the named ops above (data_out_XXX), not the prev ops (o_XXX)
        # otherwise when I freeze the graph up to this point, the named versions will be cut
        # eos is diff size to others, so excluding that
        data_out_mdn = tf.identity([data_out_pi,
                                    data_out_mu1,
                                    data_out_mu2,
                                    data_out_sigma1,
                                    data_out_sigma2,
                                    data_out_corr],
                                   name="data_out_mdn")

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.eos = o_eos

        lossfunc = get_lossfunc(
            o_pi,
            o_mu1,
            o_mu2,
            o_sigma1,
            o_sigma2,
            o_corr,
            o_eos,
            x1_data,
            x2_data,
            eos_data)
        self.cost = lossfunc / (args.batch_size * args.seq_length)

        self.train_loss_summary = tf.summary.scalar('train_loss', self.cost)
        self.valid_loss_summary = tf.summary.scalar(
            'validation_loss', self.cost)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, num=1200):

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print('error with sampling ensemble')
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
        prev_state = sess.run(self.cell.zero_state(1, tf.float32))

        strokes = np.zeros((num, 3), dtype=np.float32)
        mixture_params = []

        for i in range(num):

            feed = {self.input_data: prev_x, self.state_in: prev_state}

            [o_pi,
             o_mu1,
             o_mu2,
             o_sigma1,
             o_sigma2,
             o_corr,
             o_eos,
             next_state] = sess.run([self.pi,
                                     self.mu1,
                                     self.mu2,
                                     self.sigma1,
                                     self.sigma2,
                                     self.corr,
                                     self.eos,
                                     self.state_out],
                                    feed)

            idx = get_pi_idx(random.random(), o_pi[0])

            eos = 1 if random.random() < o_eos[0][0] else 0

            next_x1, next_x2 = sample_gaussian_2d(
                o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

            strokes[i, :] = [next_x1, next_x2, eos]

            params = [
                o_pi[0],
                o_mu1[0],
                o_mu2[0],
                o_sigma1[0],
                o_sigma2[0],
                o_corr[0],
                o_eos[0]]
            mixture_params.append(params)

            prev_x = np.zeros((1, 1, 3), dtype=np.float32)
            prev_x[0][0] = np.array([next_x1, next_x2, eos], dtype=np.float32)
            prev_state = next_state

        strokes[:, 0:2] *= self.args.data_scale
        return strokes, mixture_params
