from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np

class Model(ABC):
    
    def __init__(self):
        
        self.sess = tf.Session()        
        self.initialize = False
        self.build()
        
    def build(self):
        
        self.build_forward()
        self.build_train()
        self.build_aux()
    
    @abstractmethod    
    def build_forward(self):
        '''Build forward pass'''
        
        pass

    @abstractmethod   
    def build_train(self):
        '''Create self.loss and self.trian_step here'''
        
        pass
        
    def build_aux(self):
        '''add other things you may want here'''
        
        # predicted probabilities
        _probs = tf.nn.softmax(self.logits)
 
        # predicted class (arg max)
        self.pred = tf.argmax(_probs, axis = 1)
        
    def train(self, X_train, y_train,num_epoch, batch_size, metric = None, initialize = False, verbose = False):
        
        if not self.initialize or initialize:
            print('Initializing Model')
            init_var = tf.global_variables_initializer()
            self.sess.run(init_var)
            self.initialize = True

        N = len(X_train)

        idx = np.arange(N)
        n = (N//batch_size)*batch_size
        loss_val = []

        for epoch in range(num_epoch):

            np.random.shuffle(idx)

            idx = idx[:n].reshape([-1, batch_size])

            for batch_idx in idx:

                x_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                feed_dict = {self.x: x_batch, self.ground_truth: y_batch}
                self.sess.run(self.train_step, feed_dict=feed_dict)

                _l = self.sess.run(self.loss, feed_dict=feed_dict)
                loss_val.append(_l)
                
            if verbose:
                _y_pred = self.predict(X_train)
                _y_train = y_train
                
                if metric:
                
                    print('Epoch {}   Train Acuracy {}'.format(epoch+1, metric(_y_train, _y_pred)))
                    
                else:
                    feed_dict = {self.x: X_train, self.ground_truth: y_train}
                    _l = self.sess.run(self.loss, feed_dict=feed_dict)
                    
                    print('Epoch {}   Train Acuracy {}'.format(epoch+1, _l))
                
        return loss_val
    
    def predict(self,X, batch = 128):
        
        y_pred = []
        
        i = 0
        while i+batch <= len(X):
            xx = X[i:i+batch]
            _y  = self.sess.run(self.pred, feed_dict = {self.x: xx})
            y_pred.extend(_y)
            
            i+=batch

        xx = X[i:]
        _y  = self.sess.run(self.pred, feed_dict = {self.x: xx})
        y_pred.extend(_y)
            
        return np.array(y_pred)

def accuracy(y, y_pred):

    return sum(y==y_pred)/len(y)