import numpy as np
import argparse
import shortestPaths

if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="MFAFGeodesicFinder", description="Mean field anti-ferromagnet static metric geodesic path finder")
    parser.add_argument("T0", help="Start temperature of path", type=float)
    parser.add_argument("h0", help="Start magnetic field of path", type=float)
    parser.add_argument("T1", help="Final temperature of path", type=float)
    parser.add_argument("h1", help="Final magnetic field of path", type=float)

    parser.add_argument("-o", "--output", required=True, help="output file", type=str)

    args = parser.parse_args()

    x0 = np.array([args.T0, args.h0])
    x1 = np.array([args.T1, args.h1])
    outfile = args.output

    # print("max memusage - beginning of program:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    geo = shortestPaths.AntiFerroGeoFinder()

    res = geo(x0, x1)
    path = res["path"]
    mindist = res["dist"]
    func = res["meta"]["sol_func"]

    np.savez(outfile, path=path, x0=x0, x1=x1, mindist = mindist, func=func)
    