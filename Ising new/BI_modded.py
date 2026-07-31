import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit

# Constants
k_B = 1.380649e-23
meV_to_J = 1.60218e-22

@njit
def delta_energy(lattice, i, j, J):
    N = lattice.shape[0]
    s = lattice[i, j]
    # Periodic boundary conditions
    neighbors = (
        lattice[(i+1)%N, j] +
        lattice[(i-1)%N, j] +
        lattice[i, (j+1)%N] +
        lattice[i, (j-1)%N]
    )
    return 2 * J * s * neighbors

@njit
def metropolis_step(lattice, T, J):
    N = lattice.shape[0]
    for _ in range(N*N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        dE = delta_energy(lattice, i, j, J)

        # Standard random calls are optimized by Numba
        if dE < 0 or np.random.random() < np.exp(-dE / (k_B * T)):
            lattice[i, j] *= -1

@njit
def simulate(N, T, J, steps=2500, equil_steps=2000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Starting with a cold start (all ones) is more stable for Ferromagnets
    lattice = np.ones((N, N), dtype=np.int8)

    # Thermalization
    for _ in range(equil_steps):
        metropolis_step(lattice, T, J)

    # Data collection
    total_mag = 0.0
    for _ in range(steps):
        metropolis_step(lattice, T, J)
        total_mag += np.abs(np.sum(lattice)) / (N * N)

    return total_mag / steps

def compute_M_vs_T(N, T_values, J):
    # Joblib still manages the parallel cores
    M_list = Parallel(n_jobs=-1)(
        delayed(simulate)(N, T, J, seed=42+i)
        for i, T in enumerate(T_values)
    )
    return np.array(M_list)

def log_likelihood(J, T, M_data, sigma, N):
    M_model = compute_M_vs_T(N, T, J)
    return -0.5 * np.sum(((M_data - M_model) / sigma)**2)

def infer_J(T, M_data, N, sigma, J_min_meV, J_max_meV, n_points=15):
    # Convert meV input to Joules for the physics loops
    J_vals_J = np.linspace(J_min_meV * meV_to_J, J_max_meV * meV_to_J, n_points)
    logL = []

    for J in J_vals_J:
        print(f"Testing J = {J/meV_to_J:.2f} meV")
        logL.append(log_likelihood(J, T, M_data, sigma, N))

    logL = np.array(logL)
    probs = np.exp(logL - np.max(logL))
    probs /= np.sum(probs)

    return J_vals_J, probs

if __name__ == "__main__":
    start_time = time.time()

    # 1. Load and Normalize Data
    data = np.loadtxt("ising_data.csv", delimiter=",", skiprows=1)
    T = data[:, 0]
    M_raw = data[:, 1]
    M_data = M_raw / np.max(M_raw) # Scale to [0, 1]

    # 2. Parameters
    N = 30           # Lattice size
    sigma = 0.04     # Estimated noise/model uncertainty
    
    # 3. Infer J (Range: 1.5 to 3.0 meV)
    J_vals, probs = infer_J(T, M_data, N, sigma, 
                            J_min_meV=11, 
                            J_max_meV=13)

    J_best = J_vals[np.argmax(probs)]
    J_best_meV = J_best / meV_to_J
    print(f"\nEstimated J: {J_best_meV:.3f} meV")

    # 4. Compute best-fit curve with higher statistics for the final plot
    M_fit = compute_M_vs_T(N, T, J_best)
    
    duration = time.time() - start_time
    print(f"\nTotal time taken: {duration:.2f} seconds")

    # 5. Plot Results
    plt.figure(figsize=(8, 5))
    plt.scatter(T, M_data, label="Experimental Data (CrI3 Bulk)", color='black', alpha=0.6)
    plt.plot(T, M_fit, label=f"Bayesian Fit (J={J_best_meV:.2f} meV)", color='red', linewidth=2)
    plt.axvline(61, color='gray', linestyle='--', label="Bulk Tc = 61K")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Normalized Magnetization |M|")
    plt.title(f"Ising Model Fit for CrI3 ($N={N}$)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()