import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os

def preprocess_mnist():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = (x_train > 127).astype(np.float32).reshape((-1, 784))
    x_test = (x_test > 127).astype(np.float32).reshape((-1, 784))

    x_train = 2 * x_train - 1
    x_test = 2 * x_test - 1
    
    return x_train, x_test

def corrupt_input(x, corruption_level=0.8):
    """
    Flip random bits in a binary input (0 -> 1, 1 -> 0) with a given corruption level.
    Args:
        x: Binary input (numpy array, shape = [batch_size, 784])
        corruption_level: Probability of flipping each bit
    Returns:
        Corrupted version of x
    """
    flip_mask = np.random.binomial(1, corruption_level, size=x.shape)
    return np.abs(x - flip_mask)  # XOR operation

def shift_mnist_images(x, shift_pixels=10, direction='right'):
    """
    Shift MNIST images left or right.
    Args:
        x: Numpy array of shape (batch_size, 784)
        shift_pixels: Number of pixels to shift
        direction: 'left' or 'right'
    Returns:
        Shifted images as a numpy array of shape (batch_size, 784)
    """
    if direction not in ['left', 'right']:
        raise ValueError("direction must be 'left' or 'right'")

    x_reshaped = x.reshape(-1, 28, 28)
    if direction == 'left':
        shifted = np.roll(x_reshaped, shift=-shift_pixels, axis=2)
    else:  # right
        shifted = np.roll(x_reshaped, shift=shift_pixels, axis=2)

    # Zero out the wrapped-around pixels
    if direction == 'left':
        shifted[:, :, -shift_pixels:] = 0
    else:  # right
        shifted[:, :, :shift_pixels] = 0

    return shifted.reshape(-1, 784)

def test_reconstruction_dual(rbm, x_test, indices=[0], cor = 0, shift = 0, save=True):
    n = len(indices)
    originals = []
    tests = []
    recon_cd = []

    for index in indices:

        original = x_test[index:index+1]
        test = corrupt_input(original, corruption_level=cor)
        test = shift_mnist_images(original, shift_pixels=shift, direction='right')

        v = tf.convert_to_tensor(test, dtype=tf.float32)

        v_cd  = rbm.gibbs_update(v)
        

        originals.append(original.reshape(28, 28))
        tests.append(test.reshape(28, 28))
        recon_cd.append((v_cd.numpy().reshape(28, 28)+1)/2)

    fig, axes = plt.subplots(3, n, figsize=(n * 2, 6))
    for i in range(n):
        axes[0, i].imshow(originals[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_ylabel("Original", fontsize=10)

        axes[1, i].imshow(tests[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_ylabel("corrupted", fontsize=10)

        axes[2, i].imshow(recon_cd[i], cmap='gray')
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_ylabel("CD", fontsize=10)

    plt.tight_layout()
    if save:
        save_dir = f'images'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"images/n{rbm.n_hidden}_{rbm.n_epochs}.png")
    else:
        plt.show()

    plt.close(fig)

