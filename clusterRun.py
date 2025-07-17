import subprocess
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Clustered MFAFGeodesicFinder", description="Clustered call to Mean field anti-ferromagnet static metric geodesic path finder")
    parser.add_argument("T0", help="initial temperature", type=float)
    parser.add_argument("h0", help="initial magnetic field", type=float)
    parser.add_argument("-f", "--folder", required=True, help="output folder", type=str)
    parser.add_argument("-m", "--mem", required=False, help="memory limit in MB", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(f"{args.folder}/sim_out", exist_ok=True)
    os.makedirs(f"{args.folder}/stderr", exist_ok=True)
    os.makedirs(f"{args.folder}/stdout", exist_ok=True)

    Tf = np.linspace(0.0001,0.9999, 301)
    hf = np.array([tc/2 * np.log((1+np.sqrt(1-tc))/(1-np.sqrt(1-tc))) + np.sqrt(1-tc) for tc in Tf]) * 0.99
    Tf *= 0.99

    cmd = lambda inp: ["python3", "antiFerroPaths.py", str(inp["T0"]), str(inp["h0"]), str(inp["T1"]), str(inp["h1"]), "-o", inp["program_output"]]
    
    cmd_cluster = lambda inp: ["bsub", "-J", inp["jobname"] , "-R", f"rusage[mem={inp["mem"]}MB]", 
                               "-o", f"{inp['outfiles']}.out", "-e", f"{inp['errfiles']}.err"] + cmd(inp)

    for (T1, h1) in zip(Tf, hf):
        inp = {
            "T0": args.T0,
            "h0": args.h0,
            "T1": T1,
            "h1": h1,
            "mem": args.mem,
            "program_output": f"{args.folder}/sim_out/T0={args.T0}_h0={args.h0}_T1={T1:.3f}_h1={h1:.3f}.npz",
            "jobname": f"T0={args.T0}_h0={args.h0}_T1={T1:.2f}_h1={h1:.3f}",
            "errfiles": f"{args.folder}/stderr/T0={args.T0}_h0={args.h0}_T1={T1:.3f}_h1={h1:.3f}.e",
            "outfiles": f"{args.folder}/stdout/T0={args.T0}_h0={args.h0}_T1={T1:.3f}_h1={h1:.3f}.o",
        }
        subprocess.run(cmd_cluster(inp))
        print(f"running job {inp['jobname']} with T1={T1:.3f} and h1={h1:.3f}")
        print(f"command: {' '.join(cmd_cluster(inp))}")
        print(f"output file: {inp['program_output']}")
        print()