import numpy as np
import pandas as pd
from joblib import Parallel, delayed

k_B = 1.380649e-23  # J/K
meV_to_J = 1.60218e-22

def create_lattice(N, rng):
    return rng.choice([-1, 1], size=(N, N))


def delta_energy(lattice, i, j, J):
    N = lattice.shape[0]
    s = lattice[i, j]
    neighbors = (
        lattice[(i+1)%N, j] +
        lattice[(i-1)%N, j] +
        lattice[i, (j+1)%N] +
        lattice[i, (j-1)%N]
    )
    return 2 * J * s * neighbors


def metropolis_step(lattice, T, J, rng):
    N = lattice.shape[0]
    for _ in range(N*N):
        i = rng.integers(0, N)  
        j = rng.integers(0, N)
        dE = delta_energy(lattice, i, j, J)

        if dE < 0 or rng.random() < np.exp(-dE / (k_B * T)):
            lattice[i, j] *= -1
    return lattice


def simulate(N, T, J, steps=1500, equil_steps=1000, seed=None):
    rng = np.random.default_rng(seed)
    lattice = create_lattice(N, rng)

    # Equilibrate
    for _ in range(equil_steps):
        metropolis_step(lattice, T, J, rng)

    mags = []
    for _ in range(steps):
        metropolis_step(lattice, T, J, rng)
        mags.append(abs(np.sum(lattice) / lattice.size))

    return np.mean(mags)


def compute_M_vs_T(N, T_values, J):
    M_list = Parallel(n_jobs=-1)(
        delayed(simulate)(N, T, J, seed=42+i)
        for i, T in enumerate(T_values)
    )
    return np.array(M_list)

def add_noise(M, sigma=0.03, seed=0):
    rng = np.random.default_rng(seed)
    noisy = M + rng.normal(0, sigma, size=len(M))
    return np.clip(noisy, 0, 1)


if __name__ == "__main__":

    # Parameters
    N = 30
    J = 12 * meV_to_J
    T_values = np.linspace(100, 400, 25)  # Kelvin
    use_noise = True

    print("Generating data...")
    
    M = compute_M_vs_T(N, T_values, J)

    if use_noise:
        M = add_noise(M)

    # Save CSV
    data = np.column_stack((T_values, M))
    np.savetxt("ising_data.csv", data, delimiter=",", header="T,M", comments="")

    print("Saved to ising_data.csv")