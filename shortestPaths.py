import itertools
import numpy as np
import scipy as sc
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm.notebook import tqdm, trange
from p_tqdm import p_map
from inspect import signature

class GeoFinder:
    def __init__(self, metric, christoffel_func):
        self.metric = metric
        self.christoffel_func = christoffel_func

    def geodesic_equation(self, t, y):
        """y = [x^i, v^i] where v^i = dx^i/dt"""
        dim = len(y) // 2
        x, v = y[:dim], y[dim:-1]
        Gamma = self.christoffel_func(x)
        dvdt = -np.einsum('ijk,j,k->i', Gamma, v, v)  # Geodesic equation
        return np.concatenate([v, dvdt, [np.sqrt(np.einsum("ij,i,j", self.metric(x), v, v))]])

    def apply_limits(self, y):
        return y

    # shooting + compartmentalizing
    def shooting_and_comp(self, x0, x1, tol=1e-2):
        """Find the initial velocity that connects x0 to x1"""
        dim = len(x0)
        straight_path = np.linspace(x0, x1, 100)
        straight_dist = np.sum([np.sqrt((x-y).T @ self.metric((x+y)/2) @ (x-y)) for x,y in zip(straight_path[1:], straight_path[:-1])])
        stopevent = lambda t, y, *args: straight_dist * 1.02 - y[-1]
        stopevent.terminal = True
        
        def objective(alpha):
            # Solve the geodesic equation with initial conditions
            y0 = np.concatenate([x0, [np.cos(alpha), np.sin(alpha)], [0]])
            sol = solve_ivp(self.geodesic_equation, (0, straight_dist*20), y0, 
                            max_step=tol*0.5, events=(stopevent, ))
            xs = sol.y[:dim, :]
            # print(np.linalg.norm(xs.T - x1, axis=1))
            
            # Return the error (distance to target)
            return np.min(np.linalg.norm(self.apply_limits((xs.T - x1).T).T, axis=1))

        def shots(objective, alphas):
            mindist = p_map(objective, alphas) #, tqdm=tqdm
            return np.argmin(mindist), min(mindist)

        alpharange = (0, 2*np.pi)
        mindist=tol+1
        N = 30
        for _ in trange(100):
            alphas = np.linspace(alpharange[0], alpharange[1], N+1)[:-1]
            dalpha = (alpharange[1] - alpharange[0])/N
            ix, mindist = shots(objective, alphas)
            alphamin = alphas[ix]
            alpharange = (alphamin-dalpha, alphamin+dalpha)
            # print(mindist, ":", np.rad2deg(alpharange), np.rad2deg(alphamin))
            if mindist<tol:
                break
        
        return alphamin, mindist
    
    def __call__(self, x0, x1, tol=1e-2):
        """Find the geodesic path from x0 to x1"""
        alpha, mindist = self.shooting_and_comp(x0, x1, tol)
        v0 = [np.cos(alpha), np.sin(alpha)]
        y0 = np.concatenate([x0, v0, [0]])
        stopevent = lambda t, y, *args: mindist*1.02 - y[-1]
        stopevent.terminal = True
        sol = solve_ivp(self.geodesic_equation, [0, 10], y0, max_step=5e-3, events=(stopevent, ))
        return {"path": sol.y[:len(x0), :], "α0": alpha, "dist": mindist}

class InformationGeoFinder(GeoFinder):
    def __init__(self, freeEnergy, dx = 1e-4, dim=None):
        self.freeEnergy = freeEnergy
        if not dim:
            self.dim = len(signature(freeEnergy).parameters) - 1
        else:
            self.dim = dim
        self.dx = dx
    
    def metric(self, x):
        """
        Compute the metric tensor at point x
        """
        assert len(x) == self.dim, "x must have the same dimension as the metric"
        metric = np.zeros((self.dim, self.dim))
        f0 = self.freeEnergy(*x)
        # Compute the non-diagonal elements
        for i,j in itertools.combinations(range(self.dim), 2):
            x1 = x + self.dx * np.eye(len(x))[i]
            x2 = x + self.dx * np.eye(len(x))[j]
            x3 = x + self.dx * np.eye(len(x))[i] + self.dx * np.eye(len(x))[j]
            f1 = self.freeEnergy(*x1)
            f2 = self.freeEnergy(*x2)
            f3 = self.freeEnergy(*x3)
            metric[i,j] = (f3 + f0 - f1 - f2) / (self.dx**2)
        # Compute the diagonal elements
        for i in range(self.dim):
            x1 = x + self.dx * np.eye(len(x))[i]
            x2 = x - self.dx * np.eye(len(x))[i]
            f1 = self.freeEnergy(*x1)
            f2 = self.freeEnergy(*x2)
            metric[i,i] = (f1 + f2 - 2* f0) / (self.dx**2)
        return metric

    def christoffel_func(self, x):
        """
        Compute the Christoffel symbols at point x
        """
        assert len(x) == self.dim, "x must have the same dimension as the metric"
        Gamma = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    x1 = x + self.dx * np.eye(len(x))[i]
                    x2 = x + self.dx * np.eye(len(x))[j]
                    x3 = x + self.dx * np.eye(len(x))[k]
                    f0 = self.freeEnergy(*x)
                    f1 = self.freeEnergy(*x1)
                    f2 = self.freeEnergy(*x2)
                    f3 = self.freeEnergy(*x3)
                    Gamma[i,j,k] = (f3 + f0 - f1 - f2) / (self.dx**2)
        return Gamma

class SphereGeoFinder(GeoFinder):
    def __init__(self):
        self.dim = 2

    def apply_limits(self, y):
        return np.array((y[0,:] % np.pi, y[1,:]%(np.pi*2)))

    def metric(self, x):
        return np.diag([1, np.sin(x[0])**2])
    
    def christoffel_func(self, x):
        Gamma = np.zeros((2, 2, 2))
        Gamma[1,0,1] = np.tan(x[0])**-1
        Gamma[1,1,0] = np.tan(x[0])**-1
        Gamma[0,1,1] = -np.cos(x[0])*np.sin(x[0])
        return Gamma

# Example usage
if __name__ == "__main__":
    # geo = InformationGeoFinder(lambda β, α: β + 
    #     np.log(np.cosh(α) + np.sqrt(np.exp(-4*α) + np.sinh(α)**2)), dx=1e-4, dim=2)

    geo = SphereGeoFinder()

    # Start and end points
    x0 = np.array([np.pi/3, 0])
    x1 = np.array([np.pi/3, 1.5*np.pi/2])
    
    res = geo(x0, x1, tol=1e-2)
    path = res["path"]
    alpha = res["α0"]
    mindist = res["dist"]
    print('Initial "angle:"', np.rad2deg(alpha))
    print('minimal distance =', mindist)

    plt.figure(figsize=(8, 6))
    plt.plot(path[0], path[1], label="Geodesic")
    plt.scatter(x0[0], x0[1], c='r', label='Start', s=100)
    plt.scatter(x1[0], x1[1], c='g', label='End', s=100)
    plt.xlabel("β")
    plt.ylabel("α")
    plt.show()
