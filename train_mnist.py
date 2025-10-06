from RBM import RBM
import utils as mnist
import time
import tensorflow as tf
def train(n_hidden, n_epochs):
    start_time = time.time()
    rbm = RBM(n_visible = x_train.shape[1], n_hidden=n_hidden, n_epochs=n_epochs, eta=10**(-4), batch_size=128)
    
    rbm, _ =rbm.fit(x_train, batch_size=64)
    
    rbm.save(f'mnist/original/n{n_hidden}_{n_epochs}')
    #rbm.save(f"mnist/shifted/n{n_hidden}_{n_epochs}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training took {elapsed_time:.2f} seconds.")

def reconstruct(n_hidden, n_epochs):
    start_time = time.time()
    
    rbm = RBM(n_visible = x_train.shape[1], n_hidden=n_hidden, n_epochs=n_epochs, eta=10**(-4), batch_size=128)
    
    rbm = rbm.load(f'mnist/original/n{n_hidden}_{n_epochs}.npz')
    rbm.n_epochs = tf.constant(n_epochs)
    rbm.save(f'mnist/original/n{n_hidden}_{n_epochs}')
    mnist.test_reconstruction_dual(rbm, x_test, indices=[0, 2, 6, 12, 24], cor = 0, shift = 0, save=True)
    
    #ising.plot_loss(rbm, save=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Reconstruction took {elapsed_time:.2f} seconds.")


n_hidden = [50,128,500,900,2048]
n_epochs = [50,100,200,500]
x_train, x_test = mnist.preprocess_mnist()

for hidden in n_hidden:
    for epochs in n_epochs:
        #train(hidden, epochs)
        reconstruct(hidden, epochs)

