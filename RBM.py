import numpy as np
import tensorflow as tf
import progressbar
import os
import matplotlib.pyplot as plt


class RBM(object):
    """Restricted Boltzmann Machine"""
    def __init__(self, n_visible=4096, n_hidden=400, W=None, hbias=None, vbias=None, n_epochs=100, eta=0.0001, batch_size=128 ,random_state=42):
        """
        W: shape = [n_hidden, n_visible]
        hbias: shape = [n_hidden, ]
        vbias: shape = [n_visible, ]
        """

        self.random = np.random.RandomState(random_state)
        self.n_visible = tf.constant(n_visible)
        self.n_hidden = tf.constant(n_hidden)
        self.n_epochs = tf.constant(n_epochs)
        self.eta = tf.constant(eta) # learning rate
        self.batch_size = tf.constant(batch_size)
        self.eval_ = {'cross_entropy':[], 'pseudo_likelihood':[], 'visible_energy':[]}


        if W is None:
            W = tf.Variable(tf.random.normal([n_hidden, n_visible],
                                             mean=0,stddev=tf.sqrt(2/float(n_hidden+n_visible))), dtype='float32')
        if hbias is None:
            hbias = tf.Variable(tf.zeros([n_hidden]), dtype='float32')
        if vbias is None:
            vbias = tf.Variable(tf.zeros([n_visible]), dtype='float32')

        self.W = tf.Variable(W)
        self.hbias = hbias
        self.vbias = vbias
        self.params = [self.W, self.hbias, self.vbias]

    def update_beta(self, beta):
        self.W = tf.Variable(beta * self.W)

    def propup(self, vis):
        """return p(h|v)
        """
        wxv_c = (tf.matmul(vis, tf.transpose(self.W)) + self.hbias)*2 # [batch_size, n_hidden] 
        return self._sigmoid(wxv_c)

    def v_to_h(self, v0_sample):
        """v -> h, sample h given v from p(h|v)
        """
        h1_mean = self.propup(v0_sample)
        h1_sample = (self.random.binomial(n=1, p=h1_mean, size=h1_mean.shape)-0.5)*2
        return tf.Variable(h1_sample,dtype='float32')

    def propdown(self, hid):
        """return p(v|h)
        """
        hxw_b = (tf.matmul(hid, self.W) + self.vbias)*2
        return self._sigmoid(hxw_b)

    def h_to_v(self, h0_sample):
        """h -> v, sample v given h from p(v|h)
        """
        v1_mean = self.propdown(h0_sample)
        v1_sample = (self.random.binomial(n=1, p=v1_mean, size=v1_mean.shape)-0.5)*2
        return tf.Variable(v1_sample,dtype='float32')

    def gibbs_update(self, v_sample, k=5):
        """gibbs sampling
        CD-k to obtain batch_size number of visible states
        """
        for step in range(k):
            h_sample = self.v_to_h(v_sample)
            v_sample = self.h_to_v(h_sample)
        return v_sample

    def fit(self, X, batch_size=None):
        
        """training of RBM
        """
        m_instances = X.shape[0]
        if batch_size is not None:
            n_batches = batch_size
        else:
            n_batches = m_instances // self.batch_size.numpy()
        bar = progressbar.ProgressBar(maxval=int(self.n_epochs.numpy()),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for epoch in range(self.n_epochs):

            for batch_index in range(n_batches):
                
                indices = np.random.randint(m_instances, size=self.batch_size) #stochastic gradient descent
                v_batch = tf.Variable(X[indices, :],dtype='float32')

                # positive phase 
                with tf.GradientTape(persistent = True) as tp:
                    loss = tf.reduce_mean(-tf.reshape(tf.matmul(v_batch, tf.reshape(self.vbias,[self.n_visible,1])), [v_batch.shape[0]]) 
                                           - tf.reduce_sum( tf.math.log(tf.math.exp(tf.clip_by_value(-(tf.matmul(v_batch,tf.transpose(self.W)) + self.hbias), -50., 50)) + 
                                                  tf.math.exp(tf.clip_by_value(tf.matmul(v_batch,tf.transpose(self.W)) + self.hbias, -50., 50))), axis = 1 ))
                dW = tp.gradient(loss, self.W)
                dhbias = tp.gradient(loss, self.hbias)
                dvbias = tp.gradient(loss, self.vbias)
                del tp

                # negative phase
                v_sample = self.gibbs_update(v_batch, k=5)
                with tf.GradientTape(persistent = True) as tp:
                    loss = tf.reduce_mean(-tf.reshape(tf.matmul(v_sample, tf.reshape(self.vbias,[self.n_visible,1])), [v_sample.shape[0]]) 
                                           - tf.reduce_sum( tf.math.log(tf.math.exp(tf.clip_by_value(-(tf.matmul(v_sample,tf.transpose(self.W)) + self.hbias), -50., 50)) + 
                                                  tf.math.exp(tf.clip_by_value(tf.matmul(v_sample,tf.transpose(self.W)) + self.hbias, -50., 50))), axis = 1 ))
                neg_dW = tp.gradient(loss, self.W)
                neg_dhbias = tp.gradient(loss, self.hbias)
                neg_dvbias = tp.gradient(loss, self.vbias)

                if neg_dW is not None:
                    dW -= neg_dW
                if neg_dhbias is not None:
                    dhbias -= neg_dhbias
                if neg_dvbias is not None:
                    dvbias -= neg_dvbias
                del tp
                
                self.W.assign_sub(self.eta * dW)
                self.hbias.assign_sub(self.eta * dhbias )     
                self.vbias.assign_sub(self.eta * dvbias)
                
            
            CE = self.cross_entropy(X)
            L = self.pseudo_likelihood(X)
            visible_energy = tf.reduce_mean(self.visible_energy(X)).numpy()
            self.eval_['cross_entropy'].append(CE)
            self.eval_['pseudo_likelihood'].append(L)
            self.eval_['visible_energy'].append(visible_energy)
            bar.update(epoch+1)
            
        bar.finish()

        return self, self.eval_

    def _sigmoid(self, z):
        """ logistic function
        """
        return 1. / (1. + tf.math.exp(-tf.clip_by_value(z, -250, 250)))

    def cross_entropy(self,v_input):
        """return mean of cross entropy
        """
        v_input = tf.Variable(v_input,dtype='float32')
        h = self.v_to_h(v_input)
        p = self.propdown(h)
        
        v_input = v_input/2+0.5
        J = v_input*tf.math.log(tf.clip_by_value(p,0.000001,1)) + (1-v_input)*tf.math.log(tf.clip_by_value(1-p,0.000001,1))
        return -tf.reduce_mean(J).numpy()
    
    def pseudo_likelihood(self,v_input):
        """return mean of pseudo-likelihood
        """
        v = []
        v_ = []
        for i in range(v_input.shape[0]):
            idx = np.random.randint(0,v_input.shape[1])
            v_copy = v_input[i].copy()
            v_copy[idx] *= (-1)
            v_.append(v_copy)

        e = self.visible_energy(tf.Variable(v_input,dtype='float32'))
        e_ = self.visible_energy(tf.Variable(v_,dtype='float32'))
        PL = -self.n_visible.numpy() * np.log(self._sigmoid(e_-e).numpy())
        return np.mean(PL)
        
    def visible_energy(self, v_sample):
        """return batch_size number of visible energy 
        """
        v_sample = tf.Variable(v_sample,dtype='float32')
        wxv_c =  tf.matmul(v_sample,tf.transpose(self.W)) + self.hbias
        vbias_term = tf.reshape(tf.matmul(v_sample, tf.reshape(self.vbias,[self.n_visible,1])), [v_sample.shape[0]])
        hidden_term = tf.reduce_sum( tf.math.log(tf.math.exp(tf.clip_by_value(-wxv_c, -50., 50)) + 
                                                 tf.math.exp(tf.clip_by_value(wxv_c, -50., 50))), axis = 1 )
        return -vbias_term-hidden_term
    
    def save(rbm, filepath):
        """Save the RBM model's full state to a single compressed .npz file."""
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        np.savez_compressed(filepath,
            W=rbm.W.numpy(),
            hbias=rbm.hbias.numpy(),
            vbias=rbm.vbias.numpy(),
            n_visible=int(rbm.n_visible.numpy()),
            n_hidden=int(rbm.n_hidden.numpy()),
            n_epochs=int(rbm.n_epochs.numpy()),
            eta=float(rbm.eta.numpy()),
            batch_size=int(rbm.batch_size.numpy()),
            eval_=rbm.eval_
        )

    @classmethod
    def load(cls, filepath):
        """Load an RBM model from a .npz file and return a new RBM instance."""
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        data = np.load(filepath, allow_pickle=True)

        rbm = cls(
            n_visible=int(data["n_visible"]),
            n_hidden=int(data["n_hidden"]),
            n_epochs=int(data["n_epochs"]),
            eta=float(data["eta"]),
            batch_size=int(data["batch_size"]),
            W=tf.Variable(data["W"], dtype=tf.float32),
            hbias=tf.Variable(data["hbias"], dtype=tf.float32),
            vbias=tf.Variable(data["vbias"], dtype=tf.float32),
        )

        rbm.eval_ = data["eval_"].item()

        print(f"RBM model loaded from {filepath}")
        return rbm
    
    def plot_weights(self):
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(self.W.numpy().flatten())), self.W.numpy().flatten(), label='Weights')
                plt.xlabel("Weight Index")
                plt.ylabel("Weight Value")
                plt.legend()
                plt.grid(True)
                plt.show()