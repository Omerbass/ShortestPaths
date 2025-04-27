import numpy as np

def save_metric_and_christoffel(metric_func, grid, filename):
    """
    Save the Riemannian metric, Christoffel symbols, and number of dimensions to a .npz file.

    Parameters:
        metric_func (function): A function that takes coordinates and returns the metric tensor.
        grid (list of numpy arrays): A list of 1D arrays representing the grid for each dimension.
        filename (str): The name of the .npz file to save the data.
    """
    # Number of dimensions
    n = len(grid)
    
    # Create a meshgrid for the coordinates
    mesh = np.meshgrid(*grid, indexing='ij')
    points = np.stack(mesh, axis=-1)
    
    # Evaluate the metric on the grid
    metric_grid = np.zeros(points.shape[:-1] + (n, n))
    for idx in np.ndindex(points.shape[:-1]):
        coords_point = tuple(points[idx])
        metric_grid[idx] = metric_func(*coords_point)
    
    # Compute the Christoffel symbols numerically
    christoffel_grid = np.zeros(points.shape[:-1] + (n, n, n))
    for idx in np.ndindex(points.shape[:-1]):
        coords_point = tuple(points[idx])
        metric = metric_func(*coords_point)
        inv_metric = np.linalg.inv(metric)
        
        # Compute partial derivatives of the metric
        partials = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    shifted_coords = list(coords_point)
                    h = 1e-5  # Small step for finite differences
                    shifted_coords[k] += h
                    metric_plus = metric_func(*shifted_coords)
                    shifted_coords[k] -= 2 * h
                    metric_minus = metric_func(*shifted_coords)
                    partials[k, i, j] = (metric_plus[i][j] - metric_minus[i][j]) / (2 * h)
        
        # Compute Christoffel symbols
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    christoffel_grid[idx][k][i][j] = 0.5 * sum(
                        inv_metric[k, l] * (partials[i, l, j] + partials[j, l, i] - partials[l, i, j])
                        for l in range(n)
                    )
    
    # Save to .npz file
    np.savez(filename, metric=metric_grid, christoffel=christoffel_grid, dimensions=n, **{f"x{i}": x for (x,i) in enumerate(grid)})



# Example usage
if __name__ == "__main__":
    # Define a 2D metric tensor as a function of coordinates (x, y)
    def metric_func(x, y):
        return [[1 + x**2, 0],
                [0, 1 + y**2]]
    
    # Define the grid for each dimension
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    
    # Save the metric and Christoffel symbols to a file
    save_metric_and_christoffel(metric_func, [x, y], "metric_data.npz")
