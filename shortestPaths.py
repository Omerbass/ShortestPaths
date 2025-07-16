import itertools
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
# from p_tqdm import p_map
from inspect import signature
from typing import Callable #, Iterable, Union
import warnings
import resource

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

class GeoFinder:
    def __init__(self, metric, christoffel_func, dim):
        self.metric = metric
        self.christoffel_func = christoffel_func
        self.dim = dim

    def geodesic_equation(self, t, y):
        """y = [x^i, v^i] where v^i = dx^i/dt
        assumes y:=(x, v) where x is the position and v is the velocity"""
        x, v = y[:self.dim], y[self.dim:]
        Gamma = self.christoffel_func(x)
        dvdt = -np.einsum('ijk,j,k->i', Gamma, v, v)
        return np.concatenate([v, dvdt])

    def geodesic_equation_add_total_length(self, t, y):
        """y = [x^i, v^i] where v^i = dx^i/dt
        assumes y:=(x, v, total_length) where x is the position, v is the velocity and total_length is the accumulated length"""
        x, v = y[:self.dim], y[self.dim:]
        step = np.concatenate([self.geodesic_equation(t, y[:-1]), [np.sqrt(np.einsum("ij,i,j", self.metric(x), v, v))]])
        # print(step)
        return step

    def apply_limits(self, y):
        return y
    
    def path(self, x0, alpha, dist, tol=1e-5, v0 = 1):
        """Find the geodesic path from x0 to x1"""
        stopevent = lambda t, y, *args: dist - y[-1] + 1e-5
        stopevent.terminal = True
        y0 = np.concatenate([x0, [v0*np.cos(alpha), v0*np.sin(alpha)], [0]])
        sol = solve_ivp(self.geodesic_equation, (0, dist*20), y0, max_step=tol*0.5, events=(stopevent, ))
        path = self.apply_limits(sol.y[:self.dim,:])
        return path

    # shooting + compartmentalizing
    def shooting_and_comp(self, x0, x1, tol=1e-2):
        """Find the initial velocity that connects x0 to x1"""
        dim = self.dim
        straight_path = np.linspace(x0, x1, 100)
        straight_dist = np.sum([np.sqrt((x-y).T @ self.metric((x+y)/2) @ (x-y)) for x,y in zip(straight_path[1:], straight_path[:-1])])
        stopevent = lambda t, y, *args: straight_dist * 1.02 - y[-1]
        stopevent.terminal = True
        
        def objective(alpha):
            # Solve the geodesic equation with initial conditions
            y0 = np.concatenate([x0, [np.cos(alpha), np.sin(alpha)], [0]])
            sol = solve_ivp(self.geodesic_equation_add_total_length, (0, straight_dist*20), y0, 
                            max_step=tol*0.5, events=(stopevent, ))
            xs = self.apply_limits(sol.y[:dim, :])
            # print(np.linalg.norm(xs.T - x1, axis=1))
            
            # Return the error (distance to target)
            return np.min(np.linalg.norm((xs.T - x1).T.T, axis=1))

        def shots(objective, alphas):
            mindist = list(map(objective, alphas)) #, tqdm=tqdm
            return np.argmin(mindist), min(mindist)

        alpharange = (0, 2*np.pi)
        # mindist=tol+1
        N = 50
        for _ in range(6):
            alphas = np.linspace(alpharange[0], alpharange[1], N+1)
            dalpha = (alpharange[1] - alpharange[0])/N
            print("alpharange:", np.rad2deg(alpharange))
            ix, mindist = shots(objective, alphas)
            alphamin = alphas[ix]
            alpharange = (alphamin-dalpha, alphamin+dalpha)
            # print(mindist, ":", np.rad2deg(alpharange), np.rad2deg(alphamin))
            if mindist<tol:
                break
    
        y0 = np.concatenate([x0, [np.cos(alphamin), np.sin(alphamin)], [0]])
        sol = solve_ivp(self.geodesic_equation_add_total_length, (0, straight_dist*20), y0, 
                        max_step=tol*0.5, events=(stopevent, ))

        ixf = np.argmin(np.linalg.norm(sol.y[:self.dim,:].T-x1, axis=1))
        geodesic_dist = sol.y[-1, ixf]
        return alphamin, geodesic_dist,{"mindist": mindist, "alpharange": alpharange, "ixf": ixf, "sol": sol}
    
    def bvpsolver(self, x0, x1, tol=None):
        t = np.linspace(0,1,1001)
        y_linear = np.linspace(np.append(x0, [0,1]),np.append(x1,[0,1]),len(t)).T
        def geodesics_for_bvp(t, y):
            dydt = np.array([self.geodesic_equation(t,y0) for y0 in y.T]).T
            return dydt
            
        bc = lambda y1, y2: np.concatenate([y1[:2] - x0, y2[:2] - x1])

        if tol:
            return sc.integrate.solve_bvp(geodesics_for_bvp, bc, t, y_linear, tol=tol)
        else:
            return sc.integrate.solve_bvp(geodesics_for_bvp, bc, t, y_linear)

    def shooting_method(self, x0, x1, tol=1e-2):
        """Find the geodesic path from x0 to x1"""
        alpha, geodesic_dist, meta = self.shooting_and_comp(x0, x1, tol)
        path = meta["sol"].y[:self.dim, :meta["ixf"]+1]

        return {"path": path, "α0": alpha, "dist": geodesic_dist, "meta": meta}
    
    def __call__(self, x0, x1, tol=None):
        """
        Find the geodesic path from x0 to x1.
        
        Parameters:
        x0 (array-like): Starting point of the path.
        y0 (array-like): Ending point of the path.
        tol (float, optional): Tolerance for the path finding. Default is None.
        Returns:
        dict: A dictionary containing the path, distance, and other metadata.
        """
        sol = self.bvpsolver(x0, x1, tol)
        path = sol.y[:self.dim, :]
        sol_func = sol.sol
        geodesic_dist = np.sum([np.sqrt((x-y).T @ self.metric((x+y)/2) @ (x-y)) for x,y in zip(path[:, 1:].T, path[:,:-1].T)])

        return {"path": path, "dist": geodesic_dist, "meta": {"sol": sol, "sol_func": sol_func}}

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
        g_TT = -(-T*(one_minus_m1_sq*np.arctanh(m1)**2 + one_minus_m2_sq*np.arctanh(m2)**2) + z*one_minus_m1_sq*one_minus_m2_sq*np.arctanh(m1)*np.arctanh(m2))/(2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_Th = -(m1*one_minus_m1_sq*(T - z*one_minus_m2_sq) + m2*one_minus_m2_sq*(T - z*one_minus_m1_sq))/(2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        g_hh = -(-T*(one_minus_m1_sq+one_minus_m2_sq) + 2*z*one_minus_m1_sq*one_minus_m2_sq)/(2*T**2 - 2*z**2*one_minus_m1_sq*one_minus_m2_sq)
        return np.array([[g_TT, g_Th], [g_Th, g_hh]])

    def inv_metric(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanm1 = np.arctanh(m1)
        atanm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1
        return -np.array([[(2*z*m1_sq_minus1*m2_sq_minus1 - T*(-m1**2 - m2**2 + 2))*
            2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)/(-(T*(-m1**3 + m1 - m2**3 + m2)
            - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)**2 + 
            (2*z*m1_sq_minus1*m2_sq_minus1 - T*(-m1**2 - m2**2 + 2))*
            (z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2 
            - T*((1-m1**2 )*atanm1**2 - m2_sq_minus1*atanm2**2))),
            -(T*(-m1**3 + m1 - m2**3 + m2) - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)*
            2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)/(-(T*(-m1**3 + m1 - m2**3 + m2) 
            - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)**2 + 
            (2*z*m1_sq_minus1*m2_sq_minus1 - T*(-m1**2 - m2**2 + 2))
            *(z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2 - 
            T*(-m1_sq_minus1*atanm1**2 - m2_sq_minus1*atanm2**2)))], 
            [ -(T*(-m1**3 + m1 - m2**3 + m2) - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)*
            2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)/(-(T*(-m1**3 + m1 - m2**3 + m2) 
            - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)**2 + 
            (2*z*m1_sq_minus1*m2_sq_minus1 + T*(m1_sq_minus1 + m2_sq_minus1))*
            (z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2 - 
            T*(-m1_sq_minus1*atanm1**2 - m2_sq_minus1*atanm2**2))),
            (z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2 - 
             T*(-m1_sq_minus1*atanm1**2 - 
                m2_sq_minus1*atanm2**2))*2*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)/
                (-(T*(-m1**3 + m1 - m2**3 + m2) - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)**2 + 
                (2*z*m1_sq_minus1*m2_sq_minus1 - T*(-m1**2 - m2**2 + 2))*
                (z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2 - 
                T*(-m1_sq_minus1*atanm1**2 - m2_sq_minus1*atanm2**2)))]])

    def christoffel_func(self, x):
        T,h = x
        z = self.z
        m1, m2 = self.get_m_sublattices(x)
        atanm1 = np.arctanh(m1)
        atanm2 = np.arctanh(m2)
        m1_sq_minus1 = m1**2 - 1
        m2_sq_minus1 = m2**2 - 1
        Γ_T_xx = [[((T*(-m1*m1_sq_minus1 - m2*m2_sq_minus1) - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)*(2*T*m1*(T + z*m2_sq_minus1)*m1_sq_minus1*atanm1**2 + 
            m1_sq_minus1*(-2*T**2 - 3*T*z*m2_sq_minus1 - z**2*m1_sq_minus1*m2_sq_minus1 + 2*z*(T + z*(m1*m2 - 1))*(m1 + m2)*m2_sq_minus1*atanm2)*atanm1+
            m2_sq_minus1*(-2*T**2 + 2*T*m2*(T + z*m1_sq_minus1)*atanm2 - 3*T*z*m1_sq_minus1 - z**2*m1_sq_minus1*m2_sq_minus1)*atanm2) + 
            (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*(T**2 + 2*T*m1*(h - m2*z) + z*m2_sq_minus1*(2*h*m1 - 3*m1**2*z + z))*
            atanm1**2 + m1_sq_minus1*(-2*T**2*h + T*z*(2*T*m2 - 3*h*m2_sq_minus1) + m2*z**3*m1_sq_minus1*m2_sq_minus1 + z**2*m2_sq_minus1*(3*T*m1 - h*m1**2 + h) + 
            2*z*m2_sq_minus1*(h*(T + z*(m1*m2 - 1))*(m1 + m2) + z*(-2*T*m1*m2 +z*(-2*m1**2*m2**2 + m1**2 + m2**2)))*atanm2)*atanm1 + 
            m2_sq_minus1*(-2*T**2*h + T*z*(2*T*m1 - 3*h*m1_sq_minus1) + T*(T**2 + 2*T*m2*(h - m1*z) + z*m1_sq_minus1*(2*h*m2 - 3*m2**2*z + z))*atanm2 + 
            m1*z**3*m1_sq_minus1*m2_sq_minus1 + z**2*m1_sq_minus1*(3*T*m2 - h*m2**2 + h))*atanm2)/T)/((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*
            (-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 + (T*(m1_sq_minus1 + m2_sq_minus1) +
            2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2))),
            -(h*(T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T**2*(3*m1**4 - 4*m1**2 + 3*m2**4 - 4*m2**2 + 2) + 2*T*z*m1_sq_minus1*m2_sq_minus1*
            (3*m1**2 + 2*m1*m2 + 3*m2**2 - 2) + 2*z**2*m1_sq_minus1*m2_sq_minus1*(-2*m1**2 + m1*m2**3 + m1*m2*(m1**2 - 2) + m2**2*(3*m1**2 - 2) + 1)) +
            (m1 + m2)*(-T**4*(m1**2 - m1*m2 + m2**2 - 1)*(3*m1**4 - 5*m1**2 + 3*m2**4 - 5*m2**2 + 4) +T**3*z*(-9*m1**6*m2_sq_minus1 + m1**5*m2*(2*m2**2 - 5) +
            m1**4*(-8*m2**4 + 36*m2**2 - 25) +2*m1**3*m2*(m2**4 - 6*m2**2 + 7) + m1**2*(-9*m2**2*(m2**4 - 4*m2**2 + 6) + 23) +
            m1*m2*(-5*m2**4 + 14*m2**2 - 10) + 9*m2**6 - 25*m2**4 + 23*m2**2 - 6) -T**2*z**2*m1_sq_minus1*m2_sq_minus1*(2*m1**5*m2 + 2*m1**4*(5*m2**2 - 3) +
            m1**3*(2*m2**3 + m2) + m1**2*(10*m2**4 - 24*m2**2 + 5) + m1*m2*(2*m2**4 + m2**2 - 4) -6*m2**4 + 5*m2**2 + 2) - 
            T*z**3*m1_sq_minus1*m2_sq_minus1*(2*m1**5*m2*m2_sq_minus1 + m1**4*(6*m2**4 + 3*m2**2 - 7) + 2*m1**3*m2*(m2**4 - 3*m2**2 + 2) + m1**2*(3*m2**4 - 22*m2**2 + 15)-
            2*m1*m2*m2_sq_minus1**2 - 7*m2**4 + 15*m2**2 - 6) - 2*z**4*m1_sq_minus1**2*m2_sq_minus1**2*(m1**2*(5*m2**2 - 3) - 3*m2**2 + 1)))/
            (T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 +
            (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2)))],
            [-(h*(T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T**2*(3*m1**4 - 4*m1**2 + 3*m2**4 - 4*m2**2 + 2) + 
            2*T*z*m1_sq_minus1*m2_sq_minus1*(3*m1**2 + 2*m1*m2 + 3*m2**2 - 2) + 2*z**2*m1_sq_minus1*m2_sq_minus1*(-2*m1**2 + m1*m2**3 + m1*m2*(m1**2 - 2) + m2**2*(3*m1**2 - 2) + 1)) + 
            (m1 + m2)*(-T**4*(m1**2 - m1*m2 + m2**2 - 1)*(3*m1**4 - 5*m1**2 + 3*m2**4 - 5*m2**2 + 4) +T**3*z*(-9*m1**6*m2_sq_minus1 + m1**5*m2*(2*m2**2 - 5) + 
            m1**4*(-8*m2**4 + 36*m2**2 - 25) + 2*m1**3*m2*(m2**4 - 6*m2**2 + 7) + m1**2*(-9*m2**2*(m2**4 - 4*m2**2 + 6) + 23) +
            m1*m2*(-5*m2**4 + 14*m2**2 - 10) + 9*m2**6 - 25*m2**4 + 23*m2**2 - 6) -T**2*z**2*m1_sq_minus1*m2_sq_minus1*(2*m1**5*m2 + 2*m1**4*(5*m2**2 - 3) + m1**3*(2*m2**3 + m2) +
            m1**2*(10*m2**4 - 24*m2**2 + 5) + m1*m2*(2*m2**4 + m2**2 - 4) - 6*m2**4 + 5*m2**2 + 2) -T*z**3*m1_sq_minus1*m2_sq_minus1*(2*m1**5*m2*m2_sq_minus1 + 
            m1**4*(6*m2**4 + 3*m2**2 - 7) +2*m1**3*m2*(m2**4 - 3*m2**2 + 2) + m1**2*(3*m2**4 - 22*m2**2 + 15) - 2*m1*m2*m2_sq_minus1**2 - 7*m2**4 + 15*m2**2 - 6)
             - 2*z**4*m1_sq_minus1**2*m2_sq_minus1**2*(m1**2*(5*m2**2 - 3) - 3*m2**2 + 1)))/
            (T*(T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 +
            (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*
            atanm1*atanm2))), (2*(m1 + m2)*(T*(-m1**3 + m1 - m2**3 + m2) - z*(m1 + m2)*m1_sq_minus1*m2_sq_minus1)*(T**2*(m1**2 - m1*m2 + m2**2 - 1) + 
            3*T*z*m1_sq_minus1*m2_sq_minus1 +2*z**2*m1_sq_minus1*m2_sq_minus1*(m1*m2 - 1)) + (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*
            m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) + 2*T**2*(h*(m1**3 - m1 + m2**3 - m2) - m1*m2*z*(m1**2 + m2**2 - 2)) + T*z*m1_sq_minus1*m2_sq_minus1*
            (6*h*(m1 + m2)-z*(3*m1**2 + 8*m1*m2 + 3*m2**2 - 2)) + 4*z**2*m1_sq_minus1*m2_sq_minus1*(h*(m1 + m2)*(m1*m2 - 1) + z*(-2*m1**2*m2**2 + m1**2 + m2**2)))/T)/
            ((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 + (T*(m1**2 + m2**2 - 2) + 
            2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2)))]]

        Γ_h_xx = [[(-(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2)*
            (2*T*m1*(T + z*m2_sq_minus1)*m1_sq_minus1*atanm1**2 + m1_sq_minus1*(-2*T**2 - 3*T*z*m2_sq_minus1 - z**2*m1_sq_minus1*m2_sq_minus1 +
            2*z*(T + z*(m1*m2 - 1))*(m1 + m2)*m2_sq_minus1*atanm2)*atanm1 + m2_sq_minus1*(-2*T**2 + 2*T*m2*(T + z*m1_sq_minus1)*atanm2 -
            3*T*z*m1_sq_minus1 - z**2*m1_sq_minus1*m2_sq_minus1)*atanm2) + (m1 + m2)*(T*(m1**2 - m1*m2 + m2**2 - 1) +
            z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*(T**2 + 2*T*m1*(h - m2*z) + z*m2_sq_minus1*(2*h*m1 - 3*m1**2*z + z))*
            atanm1**2 + m1_sq_minus1*(-2*T**2*h + T*z*(2*T*m2 - 3*h*m2_sq_minus1) + m2*z**3*m1_sq_minus1*m2_sq_minus1 +
            z**2*m2_sq_minus1*(3*T*m1 - h*m1**2 + h) + 2*z*m2_sq_minus1*(h*(T + z*(m1*m2 - 1))*(m1 + m2) +
            z*(-2*T*m1*m2 + z*(-2*m1**2*m2**2 + m1**2 + m2**2)))*atanm2)*atanm1 + m2_sq_minus1*(-2*T**2*h +
            T*z*(2*T*m1 - 3*h*m1_sq_minus1) + T*(T**2 + 2*T*m2*(h - m1*z) + z*m1_sq_minus1*(2*h*m2 - 3*m2**2*z + z))*atanm2 +
            m1*z**3*m1_sq_minus1*m2_sq_minus1 + z**2*m1_sq_minus1*(3*T*m2 - h*m2**2 + h))*atanm2)/T)/
            ((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) +z*m1_sq_minus1*m2_sq_minus1)**2 + 
            (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + 
            z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2))),
            ((T**2*(3*m1**4 - 4*m1**2 + 3*m2**4 - 4*m2**2 + 2) + 2*T*z*m1_sq_minus1*m2_sq_minus1*(3*m1**2 + 2*m1*m2 + 3*m2**2 - 2) +
            2*z**2*m1_sq_minus1*m2_sq_minus1*(m1**3*m2 + m1**2*(3*m2**2 - 2) + m1*m2*(m2**2 - 2) - 2*m2**2 + 1))*(T*m1_sq_minus1*atanm1**2 +
            T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2) +
            (m1 + m2)*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)*(T**2*(T*(-m1**3 + m1 - m2**3 + m2) + h*(-3*m1**4 + 4*m1**2 - 3*m2**4 + 4*m2**2 - 2)) +
            T*z*(T*(m1*m2*(3*m1**3 - 4*m1 + 3*m2**3 - 4*m2) + m1 + m2) - 2*h*m1_sq_minus1*m2_sq_minus1*(3*m1**2 + 2*m1*m2 + 3*m2**2 - 2)) +
            z**3*(m1 - 1)*(m1 + 1)*(m1 + m2)*(m2 - 1)*(m2 + 1)*(m1**2*(5*m2**2 - 3) - 3*m2**2 + 1) -
            z**2*m1_sq_minus1*m2_sq_minus1*(-T*(m1 + m2)*(4*m1**2 + m1*m2 + 4*m2**2 - 3) + 2*h*(m1**3*m2 + m1**2*(3*m2**2 - 2) + m1*m2*(m2**2 - 2) - 2*m2**2 + 1)))/T)/
            ((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 +
            (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 +
            T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2)))], 
            [((T**2*(3*m1**4 - 4*m1**2 + 3*m2**4 - 4*m2**2 + 2) + 2*T*z*m1_sq_minus1*m2_sq_minus1*(3*m1**2 + 2*m1*m2 + 3*m2**2 - 2) +
            2*z**2*m1_sq_minus1*m2_sq_minus1*(m1**3*m2 + m1**2*(3*m2**2 - 2) + m1*m2*(m2**2 - 2) - 2*m2**2 + 1))*(T*m1_sq_minus1*atanm1**2 + 
            T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2) + 
            (m1 + m2)*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)*(T**2*(T*(-m1**3 + m1 - m2**3 + m2) + h*(-3*m1**4 + 4*m1**2 - 3*m2**4 + 4*m2**2 - 2)) +
            T*z*(T*(m1*m2*(3*m1**3 - 4*m1 + 3*m2**3 - 4*m2) + m1 + m2) - 2*h*m1_sq_minus1*m2_sq_minus1*(3*m1**2 + 2*m1*m2 + 3*m2**2 - 2)) +
            z**3*(m1 - 1)*(m1 + 1)*(m1 + m2)*(m2 - 1)*(m2 + 1)*(m1**2*(5*m2**2 - 3) - 3*m2**2 + 1) -
            z**2*m1_sq_minus1*m2_sq_minus1*(-T*(m1 + m2)*(4*m1**2 + m1*m2 + 4*m2**2 - 3) + 2*h*(m1**3*m2 + m1**2*(3*m2**2 - 2) + m1*m2*(m2**2 - 2) - 2*m2**2 + 1)))/T)/
            ((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*(-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) +
            z*m1_sq_minus1*m2_sq_minus1)**2 + (T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*(T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 +
            z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2))), 
            (m1 + m2)*(-2*(T**2*(m1**2 - m1*m2 + m2**2 - 1) + 3*T*z*m1_sq_minus1*m2_sq_minus1 + 2*z**2*m1_sq_minus1*m2_sq_minus1*(m1*m2 - 1))*(T*m1_sq_minus1*atanm1**2 +
            T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2) +
            (T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)*(T**3*(m1**2 + m2**2 - 2) + 2*T**2*(h*(m1**3 - m1 + m2**3 - m2) -
            m1*m2*z*(m1**2 + m2**2 - 2)) +T*z*m1_sq_minus1*m2_sq_minus1*(6*h*(m1 + m2) - z*(3*m1**2 + 8*m1*m2 + 3*m2**2 - 2)) +
            4*z**2*m1_sq_minus1*m2_sq_minus1*(h*(m1 + m2)*(m1*m2 - 1) + z*(-2*m1**2*m2**2 + m1**2 + m2**2)))/T)/((T**2 - z**2*m1_sq_minus1*m2_sq_minus1)*
            (-(m1 + m2)**2*(T*(m1**2 - m1*m2 + m2**2 - 1) + z*m1_sq_minus1*m2_sq_minus1)**2 +(T*(m1**2 + m2**2 - 2) + 2*z*m1_sq_minus1*m2_sq_minus1)*
            (T*m1_sq_minus1*atanm1**2 + T*m2_sq_minus1*atanm2**2 + z*m1_sq_minus1*m2_sq_minus1*atanm1*atanm2)))]]

        # print("Γ_T_xx:", Γ_T_xx, "\nΓ_h_xx:", Γ_h_xx)

        return -np.array([Γ_T_xx, Γ_h_xx])


# Example usage
if __name__ == "__main__":
    print("max memusage - beginning of program:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    
    # geo = SphereGeoFinder()
    
    # x0 = np.array([0.2,0.])
    # x1 = np.array([0.2, np.pi/3])
    
    geo = AntiFerroGeoFinder()
    
    x0 = np.array([0.5,-0.1])
    x1 = np.array([(tc := 0.5), tc/2 * np.log((1+np.sqrt(1-tc))/(1-np.sqrt(1-tc))) + np.sqrt(1-tc)])
    

    # path = geo.path(x0, np.deg2rad(-30), 5000, v0 = 1000)

    res = geo(x0, x1, tol=1e-2)
    path = res["path"]
    alpha = res["α0"]
    mindist = res["dist"]
    print('Initial "angle:"', np.rad2deg(alpha))
    print('minimal distance =', mindist)

    # res = geo.bvpsolver(x0, x1)
    # path = res.y[:2,:]
    pathlength = np.sum([np.sqrt((x-y).T @ geo.metric((x+y)/2) @ (x-y)) for x,y in zip(path[:, 1:].T, path[:,:-1].T)])
    print("path length =", pathlength)

    plt.figure(figsize=(8, 6))
    plt.plot(path[0], path[1], label="Geodesic")
    plt.scatter(x0[0], x0[1], c='r', label='Start', s=100)
    plt.scatter(x1[0], x1[1], c='g', label='End', s=100)
    tc = np.linspace(0.0001, 1, 100)
    mc = np.sqrt(1-tc)
    plt.plot(tc, tc/2 * np.log((1+mc)/(1-mc)) + mc, "k")
    plt.plot(tc, -tc/2 * np.log((1+mc)/(1-mc)) - mc, "k")
    plt.xlabel("T")
    plt.ylabel("h")
    plt.show()

    print("max memusage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
