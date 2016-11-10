import theano
import theano.tensor as T
import lasagne
import numpy as np
from lasagne.regularization import l2

class FF(object):
    """
    A general FeedForward neural network
    learning_rate
    drop_out: drop out rate
    Layers: number of layers
    N_hidden: number of nodes in each hidden layer
    D_input: dimension of input layer
    D_out: dimension of output layer
    Task_type: 'regression' or 'classification'
    L2_lambda: l2 regularization
    fixlayer: a list that points out which layers do not update weights during training
              e.g. [1,3] means the the weights W(0->1) and W(2->3) do not update weights, where 0 means input layer
    """
    def __init__(self, learning_rate, drop_out, Layers, N_hidden, D_input, D_out, Task_type='regression', L2_lambda=0.0, _EPSILON=1e-12, fixlayer=[], mid_target='0'):
        #------varibles------
        #label
        self.hard_target = T.matrix('hard_target')
        #input layer
        self.l_in = lasagne.layers.InputLayer(shape=(None, D_input))
        #last hidden layer
        self.l_hid=self.l_in
        #stack hidden layers
        #l2 regularization
        self.l2_penalty = 0
        self.lr = theano.shared(np.array(learning_rate, dtype=theano.config.floatX))
        for i in range(Layers):
            self.l_hid= lasagne.layers.DenseLayer(
                self.l_hid,
                num_units=N_hidden, W=lasagne.init.HeUniform(gain='relu'),b=lasagne.init.Constant(0.001),
                nonlinearity=lasagne.nonlinearities.rectify)
            print('Add Dense layer')
            self.l2_penalty += lasagne.regularization.regularize_layer_params(self.l_hid, l2) * L2_lambda
            self.l_hid=lasagne.layers.dropout(self.l_hid,drop_out)
            print('Add Dropout layer')
        #out_layer
        if mid_target=="mid_target":
            self.l_out=lasagne.layers.DenseLayer(self.l_hid, num_units=D_out, nonlinearity=lasagne.nonlinearities.rectify)
            print('relu out')
        else:
            self.l_out=lasagne.layers.DenseLayer(self.l_hid, num_units=D_out, nonlinearity=lasagne.nonlinearities.linear)
            print('linear out')
        #select weights not to be updated
        d=1 # how many have deleted
        self.all_params = lasagne.layers.get_all_params(self.l_out)
        self.get_weights = lasagne.layers.get_all_param_values(self.l_out)
        for f in fixlayer:
            del self.all_params[(f-d)*2]
            del self.all_params[(f-d)*2]
            d+=1
        #------training function------
        #output of net for train / eval
        self.l_out_train = lasagne.layers.get_output(self.l_out, deterministic=False)
        self.l_out_eval = lasagne.layers.get_output(self.l_out, deterministic=True)
        if Task_type!='regression':
            self.l_out_train = T.exp(self.l_out_train)/T.sum(T.exp(self.l_out_train),axis=1, keepdims=True)
            self.l_out_eval = T.exp(self.l_out_eval)/T.sum(T.exp(self.l_out_eval),axis=1, keepdims=True)
            print('Add Softmax output layer')
            self.l_out_train = T.clip(self.l_out_train, _EPSILON, 1.0 - _EPSILON)
            self.l_out_eval = T.clip(self.l_out_eval, _EPSILON, 1.0 - _EPSILON)
        #loss function for train / eval
        if Task_type!='regression':
            self.loss_train = T.mean(lasagne.objectives.categorical_crossentropy(self.l_out_train, self.hard_target))
            self.loss_eval = T.mean(lasagne.objectives.categorical_crossentropy(self.l_out_eval, self.hard_target))
        else: 
            self.loss_train = T.mean(lasagne.objectives.squared_error(self.l_out_train, self.hard_target))
            self.loss_eval = T.mean(lasagne.objectives.squared_error(self.l_out_train, self.hard_target))
        self.acc = T.mean(lasagne.objectives.categorical_accuracy(self.l_out_eval, self.hard_target))
        
        #eval functions
        self.get_acc = theano.function([self.l_in.input_var, self.hard_target], self.acc)
        self.get_loss = theano.function([self.l_in.input_var, self.hard_target], self.loss_eval)
        self.updates = lasagne.updates.adam(self.loss_train + self.l2_penalty , self.all_params, learning_rate=self.lr)
        
        #train function
        self.train = theano.function([self.l_in.input_var, self.hard_target], updates=self.updates)
        self.train_loss_acc = theano.function([self.l_in.input_var, self.hard_target], [self.loss_eval, self.acc], updates=self.updates)
        #output function
        self.get_out = theano.function([self.l_in.input_var], self.l_out_eval)
        self.hid_out = theano.function([self.l_in.input_var], lasagne.layers.get_output(self.l_hid, deterministic=True))
    def saver(self,fpath):
        np.save(fpath,lasagne.layers.get_all_param_values(self.l_out))
    def loader(self,weights):
        lasagne.layers.set_all_param_values(self.l_out, weights)
