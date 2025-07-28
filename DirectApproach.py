# import itertools
import numpy as np
import scipy as sc
# from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
# from p_tqdm import p_map
# from inspect import signature
from typing import Callable #, Iterable, Union
import warnings
import resource
from boltons.debugutils import pdb_on_signal
from rich.console import Console

def fixed_point(func:Callable, start:float, args=(), xtol:float=1e-9, maxiter:int=5000, method="iteration"):
    """
    Find the fixed point of a function using iteration.

    Parameters:
    func (callable): The function for which to find the fixed point.
    start (float): The initial guess for the fixed point.
    args (iterable): arguments to be passed to function.
    xtol (float, optional): The tolerance for convergence. Default is 1e-9.
    max_iter (int, optional): The maximum number of iterations. Default is 5000.

    Returns:
    float: The fixed point of the function.
    """
    if method != "iteration":
        raise NotImplementedError
    x0 = start
    for itr in range(maxiter):
        x1 = func(x0, *args)
        if np.max(abs(x1 - x0)) < xtol:
            break
        if np.any(np.isnan(x1)):
            raise Exception(f"got NaN in fixed point. last x was {x0}, args={args}")
        x0 = x1
    if itr == maxiter - 1:
        warnings.warn(f"Tolerance not reached\nachieved tolerance = {np.max(abs(func(x0, *args) - x0)):.3e} >= {xtol} = required tolerance")
    return x1

console = Console()

class GeoFinder:
    def __init__(self, metric, d_metric, dim):
        self.metric = metric
        self.d_metric = d_metric
        self.dim = dim

    def dist(self, path):
        """
        Compute the distance of a path.

        Parameters:
        path (array-like): The path. Assumed to be dense enough.

        Returns:
        float: The distance of the path.
        """
        metric = [self.metric(x) for x in (path[1:, :] + path[:-1, :]) / 2]
        diffs = np.diff(path, axis=0)
        return np.sum([np.sqrt(d.T @ m @ d) for d,m in zip(diffs, metric)])

    def grad(self, path):
        """
        Compute the gradient of the geodesic path.

        Parameters:
        path (array-like): The geodesic path.

        Returns:
        array: The gradient of the path.
        """
        metric = [self.metric(x) for x in (path[1:, :] + path[:-1, :]) / 2]
        d_metric = [self.d_metric(x) for x in (path[1:, :] + path[:-1, :]) / 2]
        diffs = np.diff(path, axis=0)
        metric_diffs = np.einsum('lij,lj->li', metric, diffs)
        diffs_metric = np.einsum('lji,lj->li', metric, diffs)
        d_metric_diffs = np.einsum('lijk,lj,lk->li', d_metric, diffs, diffs)

        grad = diffs_metric[:-1] - metric_diffs[1:] + 0.5 * d_metric_diffs[:-1] + 0.5 * d_metric_diffs[1:]
        
        return np.concatenate((np.zeros((1, self.dim)), grad, np.zeros((1, self.dim))), axis=0)
    
    def shortestpath(self, x0, x1, tol=None, N=100, delta = None):
        """
        Find the shortest path between two points using the geodesic metric.

        Parameters:
        x0 (array-like): Starting point of the path.
        x1 (array-like): Ending point of the path.
        tol (float, optional): Tolerance for the path finding. Default is None.

        Returns:
        dict: A dictionary containing the path, distance, and other metadata.
        """
        if tol is None:
            tol = 2e-5
        if delta is None:
            delta = np.linalg.norm(x1-x0) / N /10
        path = np.linspace(x0, x1, N)
        d0 = self.dist(path)
        d1 = d0 + 2 * tol
        counter = 0
        with console.status("[bold green]Finding shortest path...[/bold green]"):
            while abs(d1 - d0) > tol:
                if counter % 20 == 0:
                    console.print("current accuracy:", abs(d1 - d0), style="yellow")
                d0 = d1
                grad = self.grad(path)
                path = path - grad * delta / np.max((1, np.max(np.abs(grad))))
                d1 = self.dist(path)
                counter += 1
        return {"path": path, "dist": d0}
    
    def __call__(self, x0, x1, tol=None, N=100):
        """
        Find the shortest (likely geodesic) path from x0 to x1.
        
        Parameters:
        x0 (array-like): Starting point of the path.
        y0 (array-like): Ending point of the path.
        tol (float, optional): Tolerance for the path finding. Default is None.
        Returns:
        dict: A dictionary containing the path, distance, and other metadata.
        """
        sol = self.shortestpath(x0, x1, tol=tol, N=N)

        return {"path": sol["path"], "dist": sol['dist']}

class AntiFerroGeoFinder(GeoFinder):
    def __init__(self):
        self.dim = 2
        self.z = 1

    #                                        x=(T,h)
    def free_energy_non_minimized(self, m_s, x, z=1):
        T,h = x
        return 0.5*(z*(m_s[0]*m_s[1]) - h*(m_s[0] + m_s[1]) + 0.5 * T *(
        sc.special.xlogy(1+m_s[0], 1+m_s[0]) + sc.special.xlogy(1-m_s[0], 1-m_s[0]) +
        sc.special.xlogy(1+m_s[1], 1+m_s[1]) + sc.special.xlogy(1-m_s[1], 1-m_s[1]) )
        )

    #                       x=(T,h) 
    def tranceqn(self, m_s, x, z=1):
        m1 = np.tanh( (x[1] - self.z * m_s[1])/x[0] )
        m2 = np.tanh( (x[1] - self.z * m_s[0])/x[0] )
        return np.array([m1, m2])

    #                           x=(T,h) 
    def get_m_sublattices(self, x, grid=500):
        assert np.nan not in x, "x contains NaN values"
        # print(x)
        M1, M2 = np.meshgrid(*np.linspace([-1+1e-3,-1+1e-3],[1-1e-3,1-1e-3],grid).T)
        f = self.free_energy_non_minimized((M1,M2), x,z=self.z)
        ix, iy = np.unravel_index(np.argmin(f), f.shape)
        m1_0, m2_0 = M1[ix, iy], M2[ix,iy]
        if m1_0 == m2_0:
            m1_0 = np.min([m1_0 + 1e-2, 0.999])
            m2_0 = np.max([m2_0 - 1e-2, -0.999])
        # print("m0:", (m1_0, m2_0))
        m_s = fixed_point(self.tranceqn, (m1_0,m2_0), args=(x,self.z), xtol=1e-13)
        return m_s

    #                x=(T,h)
    def metric(self, x):
        z = self.z
        T,h = x
        m1, m2 = self.get_m_sublattices(x)
        one_minus_m1_sq = 1 - m1**2
        one_minus_m2_sq = 1 - m2**2
        

        g_TT = (T*((one_minus_m1_sq)*np.arctanh(m1)**2 + (one_minus_m2_sq)*np.arctanh(m2)**2) +
            2*z*(one_minus_m1_sq)*(one_minus_m2_sq)*np.arctanh(m1)*np.arctanh(m2))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_Th = (-(T + z*one_minus_m1_sq)*one_minus_m2_sq*np.arctanh(m2) - (T + z*one_minus_m2_sq)*one_minus_m1_sq*np.arctanh(m1))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_hh = (T*(-m1**2 - m2**2 + 2) + 2*z*(one_minus_m1_sq)*(one_minus_m2_sq))/\
            (2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        return np.array([[g_TT, g_Th], [g_Th, g_hh]])

    def inv_metric(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanhm1 = np.arctanh(m1)
        atanhm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1

        sec_diag = -2*(T - m2_sq_minus1*z)*atanhm1/m2_sq_minus1 - 2*(T - (m1**2 -1)*z)*atanhm2/m1_sq_minus1
        return np.array([[-2*T*(m1_sq_minus1 + m2_sq_minus1)/(m1_sq_minus1*m2_sq_minus1) + 4*z, 
                           sec_diag], 
                          [sec_diag, 
                           -2*T*atanhm1**2/m2_sq_minus1 - 2*T*atanhm2**2/m1_sq_minus1 + 4*z*atanhm1*atanhm2]]) /(atanhm1 - atanhm2)**2

    def d_metric(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanhm1 = np.arctanh(m1)
        atanhm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1

        dgdT_semi_diag = (T*m1_sq_minus1*(T**3*m1 - T**2*m1*z*m2_sq_minus1 + T*m2*z**2*m1_sq_minus1*m2_sq_minus1 - m2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + m1_sq_minus1*(-T**4 + 2*T**3*z*m2_sq_minus1 - 
            2*T**2*z*(m1 + m2)*m2_sq_minus1*(T - m1*m2*z + z)*atanhm2 - 2*T*z**3*m1_sq_minus1*m2_sq_minus1**2 + z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm1 + m2_sq_minus1*(-T**4 + 2*T**3*z*m1_sq_minus1 - 
            2*T*z**3*m1_sq_minus1**2*m2_sq_minus1 + T*(T**3*m2 - T**2*m2*z*m1_sq_minus1 + T*m1*z**2*m1_sq_minus1*m2_sq_minus1 - m1*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2 + z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm2)/(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3
        dgdh_semi_diag = -(-T**4*(m1**2 + m2**2 - 2) + 4*T**3*z*m1_sq_minus1*m2_sq_minus1 - 4*T*z**3*m1_sq_minus1**2*m2_sq_minus1**2 + 2*T*m1_sq_minus1*(T**3*m1 - T**2*z*(2*m1 + m2)*m2_sq_minus1 + 
            T*z**2*m2_sq_minus1*(m1*(2*m1*m2 + m2**2 - 1) - 2*m2) - m2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1 + 2*T*m2_sq_minus1*(T**3*m2 - T**2*z*(m1 + 2*m2)*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1*(m2*(m1 + 2*m2) - 2) - m2) - 
            m1*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2 + z**4*m1_sq_minus1**2*m2_sq_minus1**2*(m1**2 + m2**2 - 2))/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3)

        dgdT = [[(6*T*z*m1_sq_minus1*m2_sq_minus1*(-2*T**2 + T*(m1*z + m2*(T - m1*m2*z))*atanhm2 + 2*z**2*m1_sq_minus1*m2_sq_minus1)*atanhm1*atanhm2 - 
            2*T*m1_sq_minus1*(T**3*m1 - m2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**3 + 3*m1_sq_minus1*(T**4 + 2*T**2*z*m2_sq_minus1*(m1*(T - m1*m2*z) + m2*z)*atanhm2 - z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm1**2 + 
            (m2 - 1)*(m2 + 1)*(-2*T**4*m2*atanhm2 + 3*T**4 + 2*T*m1*z**3*m1_sq_minus1*m2_sq_minus1**2*atanhm2 - 3*z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm2**2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3),
          dgdT_semi_diag],[dgdT_semi_diag,
          -(-T**4*(m1**2 + m2**2 - 2) + 4*T**3*z*m1_sq_minus1*m2_sq_minus1 - 4*T*z**3*m1_sq_minus1**2*m2_sq_minus1**2 + 
          2*T*m1_sq_minus1*(T**3*m1 - T**2*z*(2*m1 + m2)*m2_sq_minus1 + T*z**2*m2_sq_minus1*(m1*(2*m1*m2 + m2**2 - 1) - 2*m2) - m2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1 + 
          2*T*m2_sq_minus1*(T**3*m2 - T**2*z*(m1 + 2*m2)*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1*(m2*(m1 + 2*m2) - 2) - m2) - m1*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2 + 
          z**4*m1_sq_minus1**2*m2_sq_minus1**2*(m1**2 + m2**2 - 2))/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3)]]

        dgdh = [[(T*m1_sq_minus1*(T**3*m1 - T**2*m1*z*m2_sq_minus1 + T*m2*z**2*m1_sq_minus1*m2_sq_minus1 - m2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + 
            m1_sq_minus1*(-T**4 + 2*T**3*z*m2_sq_minus1 - 2*T**2*z*(m1 + m2)*m2_sq_minus1*(T - m1*m2*z + z)*atanhm2 - 2*T*z**3*m1_sq_minus1*m2_sq_minus1**2 + z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm1 +
            m2_sq_minus1*(-T**4 + 2*T**3*z*m1_sq_minus1 - 2*T*z**3*m1_sq_minus1**2*m2_sq_minus1 + T*(T**3*m2 - T**2*m2*z*m1_sq_minus1 + T*m1*z**2*m1_sq_minus1*m2_sq_minus1 - m1*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2 + 
            z**4*m1_sq_minus1**2*m2_sq_minus1**2)*atanhm2)/(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3,

            dgdh_semi_diag],[dgdh_semi_diag,
            
            -T*(m1 + m2)*(-T**3*(m1**2 - m1*m2 + m2**2 - 1) + 3*T**2*z*m1_sq_minus1*m2_sq_minus1 - 3*T*z**2*m1_sq_minus1*m2_sq_minus1*(m1*m2 - 1) + 
            z**3*m1_sq_minus1*m2_sq_minus1*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))/(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3]]
        
        return np.array([dgdT, dgdh])

    def christoffel_func(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanhm1 = np.arctanh(m1)
        atanhm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1

        Γ_T_xx = [[(2*T*(T**2*m1 - m2*z**2*m1_sq_minus1**2)*atanhm1**3 + (-T**3*(3*m1**2 + m2**2 - 4)/m1_sq_minus1 + T*z**2*(m1_sq_minus1 - m2_sq_minus1)*m2_sq_minus1 + 
                    2*T*(T**2*m2 - m1*z**2*m2_sq_minus1**2)*atanhm2 + 4*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                    (-T**3*(m1**2 + 3*m2**2 - 4)/m2_sq_minus1 - T*z**2*(m1_sq_minus1 - m2_sq_minus1)*m1_sq_minus1 + 
                     2*T*(-T**2*m1 + 2*T*z*(m1 - m2)*(m1*m2 + 1) + m2*z**2*m1_sq_minus1**2)*atanhm2 + 
                     4*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2 + 2*(2*T**3 + 3*T**2*z*(m1_sq_minus1 + m2_sq_minus1) - 6*T*z**2*m1_sq_minus1*m2_sq_minus1 + 
                    T*(-T**2*m2 + 2*T*z*(-m1 + m2)*(m1*m2 + 1) + m1*z**2*m2_sq_minus1**2)*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm1*atanhm2)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2), 
                   
          (-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - 
           (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m1_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - 
            2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
            (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + 5*m2_sq_minus1) + 
                2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/
                (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2)],
        
                  [(-2*T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**2*m1 + T*z*(m1 - m2)*(m1*m2 + 1) - m2*z**2*m1_sq_minus1**2)*atanhm1**2 - (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*
                    (T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m1_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - 
                    2*T*m1_sq_minus1*(T**2*m2 + T*z*(-m1 + m2)*(m1*m2 + 1) - m1*z**2*m2_sq_minus1**2)*atanhm2 + z**3*m1_sq_minus1**2*m2_sq_minus1*(m1**2 - 3*m2**2 + 2))*atanhm2/m1_sq_minus1 + 
                    (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1_sq_minus1 + m2_sq_minus1) - 3*T**2*z*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*m2_sq_minus1*(5*m2_sq_minus1 + m1_sq_minus1) + 
                    2*T*(T**2 - z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*(m1 + m2)*m2_sq_minus1*atanhm2 - z**3*m1_sq_minus1*m2_sq_minus1**2*(3*m1**2 - m2**2 - 2))*atanhm1/m2_sq_minus1)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),

                   -(T**3*(m1_sq_minus1 + m2_sq_minus1)**2/(m1_sq_minus1*m2_sq_minus1) - 6*T**2*z*(m1_sq_minus1 + m2_sq_minus1) + T*z**2*(m1**4 + 2*m1**2*(5*m2_sq_minus1 - 1) + m2**4 - 12*m2**2 + 12) +
                      2*T*(m1 - m2)*(-atanhm1 + atanhm2)*(T**2 + 2*T*z*(m1*m2 + 1) - m1*m2*z**2*(m1**2 + m1*m2 + m2**2 - 2) + z**2) - 2*z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))/
                      (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        
        Γ_h_xx = [[(-2*T*m2*z**2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*m1_sq_minus1**2*atanhm1**4 + 
                    (-T**2 + z**2*m1_sq_minus1*m2_sq_minus1)*m2_sq_minus1*(T**3 + T**2*z*m1_sq_minus1 + 2*T*m1*z**2*m1_sq_minus1*m2_sq_minus1*atanhm2 +
                    T*z**2*m1_sq_minus1*m2_sq_minus1 - 3*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm2**3/m1_sq_minus1 + (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*
                    (-T**3 + 7*T**2*z*m2_sq_minus1 - 5*T*z**2*m1_sq_minus1*m2_sq_minus1 + 2*T*(T**2*m2 + 2*T*m1*z*m2_sq_minus1 + m1*z**2*m2_sq_minus1**2)*atanhm2 - 
                    z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm1*atanhm2**2 - (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(T**3 - 7*T**2*z*m1_sq_minus1 + 
                    2*T**2*(m1 + m2)*(T + 2*m1*m2*z - 2*z)*atanhm2 + 5*T*z**2*m1_sq_minus1*m2_sq_minus1 + z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2*atanhm2 - 
                    (T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-2*T*m2_sq_minus1*(T**2*m1 + 2*T*m2*z*m1_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 + 
                    m1_sq_minus1*(T**3 + T**2*z*m2_sq_minus1 + T*z**2*m1_sq_minus1*m2_sq_minus1 - 3*z**3*m1_sq_minus1*m2_sq_minus1**2))*atanhm1**3/m2_sq_minus1)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**3*(atanhm1 - atanhm2)**2),
                   
                   (-2*T*m2*z*m1_sq_minus1*(T - m1**2*z + z)*atanhm1**3 + 2*T*(2*(T - m1**2*z + z)*(T - m2**2*z + z) +
                   (T**2*(m1 - m2) + T*m2*z*m1_sq_minus1 - m1*z**2*m2_sq_minus1**2)*atanhm2)* atanhm1*atanhm2 + 
                    (T**3*(-m1**2 + m2**2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*m2_sq_minus1*(T - m2**2*z + z)*atanhm2 + 
                     T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) - 2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                    (T**3*(m1_sq_minus1-m2_sq_minus1)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) - 
                     2*T*(T**2*(m1 - m2) - T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 - 
                     2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2)/
                    (2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)],
                  
                  [(-2*T*m2*z*m1_sq_minus1*(T - m1**2*z + z)*atanhm1**3 + 2*T*(2*(T - m1**2*z + z)*(T - m2**2*z + z) +
                   (T**2*(m1 - m2) + T*m2*z*m1_sq_minus1 - m1*z**2*m2_sq_minus1**2)*atanhm2)*atanhm1*atanhm2 + 
                   (T**3*(-m1**2 + m2**2)/m1_sq_minus1 - 2*T**2*z*m2_sq_minus1 - 2*T*m1*z*m2_sq_minus1*(T - m2**2*z + z)*atanhm2 + T*z**2*m2_sq_minus1*(3*m1**2 + m2**2 - 4) - 
                    2*z**3*m1_sq_minus1*m2_sq_minus1**2)*atanhm2**2 + 
                   (T**3*(m1_sq_minus1-m2_sq_minus1)/m2_sq_minus1 - 2*T**2*z*m1_sq_minus1 + T*z**2*m1_sq_minus1*(m1**2 + 3*m2**2 - 4) - 
                    2*T*(T**2*(m1 - m2) - T*m1*z*m2_sq_minus1 + m2*z**2*m1_sq_minus1**2)*atanhm2 - 
                    2*z**3*m1_sq_minus1**2*m2_sq_minus1)*atanhm1**2)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2),
                   
                   -(2*T*m2*(T - m1**2*z + z)**2*atanhm1**2 + (T**3*(m1_sq_minus1 + m2_sq_minus1)/m1_sq_minus1 - T**2*z*(5*m2_sq_minus1 + m1_sq_minus1) + 2*T*m1*(T - m2**2*z + z)**2*atanhm2 + 
                    T*z**2*m2_sq_minus1*(5*m1_sq_minus1 + m2_sq_minus1) - z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm2 + (T**3*(m1_sq_minus1 + m2_sq_minus1)/m2_sq_minus1 - 
                     T**2*z*(5*m1_sq_minus1 + m2_sq_minus1) + T*z**2*m1_sq_minus1*(5*m2_sq_minus1 + m1_sq_minus1) - 2*T*(m1 + m2)*(T**2 - 2*T*z*(m1*m2 - 1) + z**2*(m1*m2*(m1**2 - m1*m2 + m2**2 - 2) + 1))*atanhm2 - 
                    z**3*m1_sq_minus1*m2_sq_minus1*(m1_sq_minus1 + m2_sq_minus1))*atanhm1)/(2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)**2*(atanhm1 - atanhm2)**2)]]
        # print("Γ_T_xx:", Γ_T_xx, "\nΓ_h_xx:", Γ_h_xx)

        return np.array([Γ_T_xx, Γ_h_xx])

if __name__ == "__main__":
    pdb_on_signal()
    geo = AntiFerroGeoFinder()
    
    x0 = np.array([0.35, 0.1])
    x1 = np.array([0.915, 0.562])

    res = geo(x0, x1)
    path = res["path"]
    mindist = res["dist"]
    print('minimal distance =', mindist)

    plt.figure(figsize=(8, 6))
    plt.plot(path[:, 0], path[:, 1], label="Geodesic")
    plt.scatter(x0[0], x0[1], c='r', label='Start', s=100)
    plt.scatter(x1[0], x1[1], c='g', label='End', s=100)
    tc = np.linspace(0.0001, 1, 200)
    mc = np.sqrt(1-tc)
    plt.plot(tc, tc/2 * np.log((1+mc)/(1-mc)) + mc, "k")
    plt.plot(tc, -tc/2 * np.log((1+mc)/(1-mc)) - mc, "k")
    plt.xlabel("T")
    plt.ylabel("h")
    plt.show()

    print("max memusage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)