import dynet as dy
import numpy as np
import json
import random

class LM(object):

    def __init__(self, num_layers, input_dim, hidden_dim, word_num, init_scale_rnn, init_scale_params, x_dropout, h_dropout, w_dropout_rate, lr, clip_thr):

        model = dy.Model()

        rnn = dy.LSTMBuilder(num_layers, input_dim, hidden_dim, model)
        self.init_rnn(rnn, init_scale_rnn)

        rnn_b = dy.LSTMBuilder(num_layers, input_dim, hidden_dim, model)
        self.init_rnn(rnn_b, init_scale_rnn)

        hidden_dim = hidden_dim*2

        params = {}
        params["embeds"] = model.add_lookup_parameters((word_num, input_dim))
        params["W_p"] = model.add_parameters((1, hidden_dim))

        if init_scale_params:
            self.init_lookup(embeds, init_scale_params)
            self.init_param(params["W_p"], init_scale_params)

        trainer = dy.SimpleSGDTrainer(model, lr)
        if clip_thr > 0:
            trainer.set_clip_threshold(clip_thr)

        self._model = model
        self._rnn = rnn
        self._rnn_b = rnn_b
        self._params = params
        self._x_dropout = x_dropout
        self._h_dropout = h_dropout
        self._w_dropout_rate = w_dropout_rate
        self._trainer = trainer
        self._input_dim = input_dim


    def init_param(self, param, scale):
        dims = param.as_array().shape
        param.set_value(2*scale*np.random.rand(*dims) - scale)

    def init_lookup(self, param, scale):
        dims = param.as_array().shape
        param.init_from_array(2*scale*np.random.rand(*dims) - scale)

    def init_rnn(self, rnn, init_scale_rnn):
        pc = rnn.param_collection()
        pl = pc.parameters_list()
        for p in pl:
            dims = p.as_array().shape
            p.set_value(2*init_scale_rnn*np.random.rand(*dims) - init_scale_rnn)
        return

    def word_dropout(self, seq, w_dropout_rate):
        w_dropout = []
        for w in set(seq):
            p = random.random()
            if p < w_dropout_rate:
                w_dropout.append(w)

        return w_dropout

    def update_lr(self, lr_decay_factor):
        self._trainer.learning_rate /= lr_decay_factor
        return

    def trainer_update(self):
        self._trainer.update()

    def get_learning_rate(self):
        return self._trainer.learning_rate

    def set_learning_rate(self, rate):
        self._trainer.learning_rate = float(rate)
        return

    def get_batch_scores(self, batch, evaluate = False):
        dy.renew_cg()

        batch_scores = []

        if evaluate:
            self._rnn.disable_dropout()
            self._rnn_b.disable_dropout()
        else:
            self._rnn.set_dropouts(self._x_dropout, self._h_dropout)
            self._rnn_b.set_dropouts(self._x_dropout, self._h_dropout)

        state = self._rnn.initial_state()
        state_b = self._rnn_b.initial_state()

        W = self._params["W_p"]

        for seq in batch:
            # use word dropout when training
            dropped = dy.inputTensor(np.zeros(self._input_dim))
            if evaluate:
                vecs = [self._params["embeds"][w] for w in seq]
            else:
                w_dropout = self.word_dropout(seq, self._w_dropout_rate)
                vecs = [self._params["embeds"][w] if w not in w_dropout else dropped for w in seq]

            outputs = [x.output() for x in state.add_inputs(vecs)]
            outputs_b = [x.output() for x in state_b.add_inputs(reversed(vecs))]
            assert(len(outputs) == len(outputs_b))

            repr = dy.concatenate([outputs[-1], outputs_b[-1]])
            score = W*repr
            batch_scores.append(score)

        return batch_scores

    def save(self, model_name, vocab=None):
        self._model.save("models/" + model_name + "_model")
        if vocab:
            json.dump(vocab, open(f"models/{model_name}_vocab", 'w'))

    def load(self, model_to_load):
        self._model.populate(model_to_load)
