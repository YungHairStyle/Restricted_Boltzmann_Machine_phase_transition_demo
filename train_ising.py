import ising
from RBM import RBM
import numpy as np
import time
import matplotlib.pyplot as plt

def train(hidden, epochs):
    x_train, x_test = ising.preprocess_ising(temp=temp, data_path="data/ising_data_complete.hdf5", test_size=0.2, seed=1)

    rbm = RBM(n_visible = x_train.shape[1], n_hidden=hidden, n_epochs=epochs, eta=10**(-4), batch_size=128, beta=beta)

    rbm, _ =rbm.fit(x_train)
    
    rbm.save(f'instances/{hidden}/rbm_T{temp}_{epochs}')

def reconstruct(hidden, epochs):
    mag = []
    x_train, x_test = ising.preprocess_ising(temp=temp, data_path="data/ising_data_complete.hdf5", test_size=0.2, seed=1)

    rbm = RBM(n_visible = x_train.shape[1], n_hidden=hidden, n_epochs=epochs, eta=10**(-4), batch_size=128)

    rbm = RBM.load(f"instances/{hidden}/rbm_T{temp}_{epochs}.npz")

    rbm.update_beta(beta)

    #rbm.plot_weights(temp)
    #ising.plot_loss(rbm, temp=temp, hidden=hidden, epochs=epochs,beta=beta)
    mag.append(ising.evaluate_magnetization(rbm, x_train, x_test, temp=temp, k=5, beta = beta, plot=False))
    return mag

def no_plot_mag(betas_inv, vals):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_hidden)))
    linestyles = ['-', '--', ':']  # Must match len(n_epochs)

    for i, hidden in enumerate(n_hidden):
        for j, epoch in enumerate(n_epochs):
            label = f"{hidden} hidden, {epoch} epochs"
            mags = vals[i][j]
            plt.plot(betas_inv, mags, label=label,
                     color=colors[i], linestyle=linestyles[j % len(linestyles)], linewidth=2)

    plt.xlabel('1 / Beta (Temperature)', fontsize=14)
    plt.ylabel('Magnetization', fontsize=14)
    plt.title('Magnetization vs 1/Beta', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    
    plt.savefig(f'figures/compare_epoch/T{temp}.svg', format='svg')
    
def plot_mag(betas_inv):
    temp_colors = plt.cm.plasma(np.linspace(0, 1, len(temps)))
    linestyles = ['-', '--', ':', '-.']  # Enough styles for hidden node configs

    for e_idx, epoch in enumerate(n_epochs):
        plt.figure(figsize=(10, 6))
        for t_idx, temp in enumerate(temps):
            for h_idx, hidden in enumerate(n_hidden):
                mags = all_vals[t_idx][h_idx][e_idx]
                label = f"T={temp}, {hidden} hidden"
                plt.plot(betas_inv, mags, label=label,
                         color=temp_colors[t_idx], linestyle=linestyles[h_idx % len(linestyles)], linewidth=2)

        plt.xlabel('1 / Beta (Temperature)', fontsize=14)
        plt.ylabel('Magnetization', fontsize=14)
        plt.title(f'Magnetization vs 1/Beta at {epoch} Epochs', fontsize=16)
        plt.legend(title="Temp & Hidden Units", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/compare_temp/epochs/e{epoch}.png', format='png')

        plt.show()

temps = [1,2.186995,3]
n_hidden = [2048,900,512,128]
n_epochs = [200,100,50]
betas = [1/6,1/4,1/3,1/2.5,1/2,1/1.5,1,2,8]


all_vals = []  # Shape: [len(temps)][len(n_hidden)][len(n_epochs)][len(betas)]

for temp in temps:
    vals_per_temp = []
    for hidden in n_hidden:
        mags_per_hidden = []
        for i, epochs in enumerate(n_epochs):
            mags = []
            for beta in betas:
                # train(hidden, epochs)  # Uncomment if training is needed per beta
                mags.append(reconstruct(hidden, epochs))  # Assuming reconstruct depends on beta
            mags_per_hidden.append(mags)
        vals_per_temp.append(mags_per_hidden)
    all_vals.append(vals_per_temp)

plot_mag(np.array([1 / beta for beta in betas]))