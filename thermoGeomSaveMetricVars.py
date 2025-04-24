from savemetricvars import save_metric_and_christoffel
import numpy as np
from scipy.special import xlogy
from tqdm import tqdm


def metric_from_free_entropy(φ, β, α, δ=1e-5):
    """
    Example metric function based on a free entropy surface.
    """

    d2φ_dα2 = (φ(β, α + δ) - 2 * φ(β, α) + φ(β, α - δ)) / (δ ** 2)
    d2φ_dβ2 = (φ(β + δ, α) - 2 * φ(β, α) + φ(β - δ, α)) / (δ ** 2)
    d2φ_dαdβ = (φ(β + δ, α + δ) - φ(β - δ, α + δ) - φ(β + δ, α - δ) + φ(β - δ, α - δ)) / (4 * δ ** 2)

    return np.array([[d2φ_dβ2, d2φ_dαdβ], [d2φ_dαdβ, d2φ_dα2]])  # Identity matrix as a simple example

def finite_antiferro_mean_field_metric(N, δ=1e-5):
    """"""
    τ = lambda x: (xlogy(1+x, 1+x) + xlogy(1-x,1-x))/2

    def phi(β, α):
        return np.log(np.sum([[np.exp((β * x * y / N + α * (x+y) - τ(x) - τ(y))/2) 
                               for x in np.linspace(-1,1, N)] for y in np.linspace(-1,1, N)]))

    return lambda β, α: metric_from_free_entropy(phi, β, α, δ=δ)

if __name__ == "__main__":
    # Define the grid for β and α
    β_grid = np.linspace(1e-3, 3, 100)
    α_grid = np.linspace(-3, 3, 101)

    # Save the metric and Christoffel symbols to a file
    for N in tqdm(np.logspace(1, 4, 5, dtype=int)):
        save_metric_and_christoffel(finite_antiferro_mean_field_metric(N),
                                    [β_grid, α_grid],
                                    f"metrics/mean_field_antiferro_finite_lattice_N={N}.npz")