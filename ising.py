import numpy as np
import tensorflow as tf
import h5py
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import os

def preprocess_ising(temp: float, data_path="data/ising_data_complete.hdf5", test_size=0.2, seed=42):
    """
    Load Ising model configurations at a specific temperature from HDF5.

    Args:
        temp: Temperature to load (e.g., 2.5)
        data_path: Relative or absolute path to the HDF5 file
        test_size: Fraction of data to reserve for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x_train, x_test), where each is a float32 np.ndarray of shape [N, L*L]
    """
    file_path = Path(data_path)

    with h5py.File(file_path, "r") as f:
        # List available temperature keys
        available_keys = list(f.keys())
        # Find closest match
        matched_key = None
        for k in available_keys:
            if np.isclose(float(k.split('=')[1]), temp, atol=1e-2):
                matched_key = k
                break

        if matched_key is None:
            raise ValueError(f"Temperature {temp} not found in file. Available keys: {available_keys}")

        configs = np.array(f[matched_key])

    # Convert spins from -1/+1 to 0/1
    configs_flat = configs.reshape(configs.shape[0], -1)

    # Train/test split
    x_train, x_test = train_test_split(
        configs_flat, test_size=test_size, random_state=seed
    )

    return x_train, x_test

def evaluate_magnetization(rbm, x_train, x_test, temp, k=2, beta = 1, plot = True):
    """
    Compute average magnetization for a batch of random test samples using RBM reconstructions.
    """
    mag = []
    mag_train = []

    # Compute magnetization for reconstructed test samples
    for i in range(500):
        v0 = tf.convert_to_tensor(x_test[i], dtype=tf.float32)
        v0 = tf.reshape(v0, [1, -1])
        data = v0

        v = rbm.gibbs_update(data, k)

        mag_train.append((np.mean(x_train[i])))
        
        v = v.numpy()
        
        mag.append((np.mean(v)))
        
    # Convert to arrays and clean invalids
    mag = np.array(mag)
    mag_train = np.array(mag_train)

    # Stats
    mean_cd = np.abs(np.mean(np.abs(mag)))
    mean_train = np.abs(np.mean(np.abs(mag_train)))
    
    print("CD mean:", mean_cd, "Train mean:", mean_train)
    # Compute relative differences
    cd_rel_diff = abs(mean_cd - mean_train) / abs(mean_train) * 100
    print(f"CD is {cd_rel_diff:.2f}% away from Training data")
    if plot:
        plot_dist(temp, [mag_train, mag], rbm.n_hidden, rbm.n_epochs, beta=beta)
    return mean_cd

def plot_dist(temp, mags, hidden, epochs,beta):
        leg=["training", "reconstructed"]
        colors=['r','g']
        plt.figure()
        for i in range(len(mags)):
            plt.scatter(np.arange(len(mags[i])), mags[i], label=leg[i], color = colors[i])
            plt.axhline(y=np.mean(mags[i]), color=colors[i], linestyle='--', label='Mean Magnetization')
            plt.axhline(y=np.mean(np.abs(mags[i])), color=colors[i], label='Absolute Magnetization')
        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Magnetization")
        plt.title("Magnetization of Training Samples")
        plt.grid(True)
        save_dir = f"figures/distributions/{beta}/{temp}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"figures/distributions/{beta}/{temp}/mag_n{hidden}_{epochs}.png")
        plt.close()

def plot_loss(rbm, temp=None, save=True):
    for metric_name, values in rbm.eval_.items():
                plt.figure(figsize=(10, 6))
                plt.scatter(range(len(values)), values, label=metric_name)
                plt.title(f"{metric_name} over epochs")
                plt.xlabel("Epoch")
                plt.ylabel(metric_name)
                plt.legend()
                plt.grid(True)
                
                if save:
                    if temp is None:
                        save_dir = f'images/original'
                    else:
                        save_dir = f"loss/{temp}"
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(save_dir + f"/{metric_name}_n{rbm.n_hidden}_{rbm.n_epochs}.png")
                
                else:
                    plt.show()
                
                plt.close()

