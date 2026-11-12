"""Generate test data for a template signal and save it to HDF5.

Python replacement for choose_template_signals.jl, same options:
    python choose_template_signals.py -i powerlaw_parameters.h5 -o powerlaw.h5 -s powerlaw
The input file holds datasets ``parameters_signal`` and ``parameters_noise`` (P, A).
"""
import argparse

import h5py
import numpy as np

from GWBackFinder import Simulator

parser = argparse.ArgumentParser()
parser.add_argument("--parameters", "-i", required=True, help="input parameters file path")
parser.add_argument("--output-dir", "-o", required=True, help="output file")
parser.add_argument("--signal", "-s", required=True,
                    choices=["powerlaw", "peak", "wiggly", "bpowerlaw", "dpeak", "three_peaks"])
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

with h5py.File(args.parameters, "r") as fh:
    params_signal = np.asarray(fh["parameters_signal"])
    params_noise = np.asarray(fh["parameters_noise"])

Data_AA, Data_TT, f_total_AA, f_total_TT = Simulator().different_signals(
    args.signal, params_signal, params_noise, rng=args.seed)

with h5py.File(args.output_dir, "w") as fh:
    fh.create_dataset("Data_AA", data=Data_AA)
    fh.create_dataset("Data_TT", data=Data_TT)
    fh.create_dataset("f_total_AA", data=f_total_AA)
    fh.create_dataset("f_total_TT", data=f_total_TT)
