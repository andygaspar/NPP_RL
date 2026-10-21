# Learning to Initialise Genetic Algorithms in Bilevel Optimisation: A Reinforcement Learning Approach

Code accompanying the paper

> A. Gasparin, F. J. Camerota Verdù, L. Castelli.
> *Learning to Initialise Genetic Algorithms in Bilevel Optimisation: A Reinforcement Learning Approach.*

This repository implements a reinforcement learning approach to the initialisation of genetic algorithms (GAs) for
bilevel optimisation problems with continuous leader variables. Rather than sampling the initial population at random,
a learned policy defines an instance-conditioned distribution over the leader's decision variables, from which the
initial population of the GA is drawn. Since the GA operators are left unchanged, the learned initialisation can be
combined with existing GA implementations.

Two bilevel problems are considered:

- **Path Pricing Problem (PP)**: the leader sets tolls on a set of paths, and each follower (commodity) selects its
  minimum-cost path. In the code, this problem is referred to as `pp`.
- **Stackelberg Knapsack Weight Problem (SKWP)**: the leader sets the weights of its items, and each follower fills its
  knapsack greedily. Both the single-follower ($K=1$) and multi-follower ($K>1$) variants are considered.

For each problem, a Graph Transformer Network (GTN) maps a graph representation of the instance to a distribution over
the leader's continuous variables. The network is trained with REINFORCE, using the quality of the solutions in the
sampled population as the reward signal. Policies are trained on small instances and applied, without retraining, to
instances many times larger than those seen during training.

## Repository structure

```
.
├── build.sh                    # compiles the two C++ libraries (see Installation)
├── requirements.txt
├── common/
│   ├── policy.py               # GTN policy (TransformerPolicy), REINFORCE update, load_policy
│   └── initialisations.py      # non-learned GA initialisations: LHC, Sobol, Halton, Kronecker
├── pp/                         # Path Pricing Problem
│   ├── instance.py             # random instance generator
│   ├── graph_instance.py       # graph representation of an instance for the GTN
│   ├── genetic_solver.py       # Python wrapper of the C++ GA
│   ├── exact_solver.py         # single-level MILP reformulation solved with Gurobi
│   ├── ga_cpp/                 # C++ GA (genetic.h, utils_.h) and its ctypes bridge
│   ├── train.py                # trains the PP policy
│   ├── validation.py           # builds the fixed validation set used for model selection
│   └── benchmark.py            # runs the experiments of the paper (GTN+GA, GA variants, Gurobi)
├── skwp/                       # Stackelberg Knapsack Weight Problem
│   ├── instance.py             # random instance generator
│   ├── graph_instance.py       # graph representation of an instance for the GTN
│   ├── ga.py                   # GA (Python), follower response computed in C++
│   ├── exact_solver.py         # MILP formulations (single and multi follower) solved with Gurobi
│   ├── knapsack_cpp/           # C++ follower response (greedy / exact knapsack) and its ctypes bridge
│   ├── train.py                # trains the SKWP policies
│   ├── validation.py           # builds the fixed validation sets used for model selection
│   └── benchmark.py            # runs the experiments of the paper
├── models/                     # trained policies (.pth)
└── results/                    # results of the paper and script to build its tables
    ├── make_tables.py
    ├── pp_results.csv  skwp_k1_results.csv  skwp_mf_results.csv
    ├── pp_exact.csv  skwp_k1_exact.csv  skwp_mf_exact.csv   # Gurobi results, read by the benchmarks
    └── validation_pp.csv  validation_skwp_k1.csv  validation_skwp_mf.csv
```

## Installation

The code was developed on Linux with Python 3 and `g++` with OpenMP support.

```bash
git clone https://github.com/OR-UniTS/rl-ga-init-bilevel.git
cd rl-ga-init-bilevel
pip install -r requirements.txt   # install the torch / torch_geometric builds that match your CUDA version
./build.sh                        # builds pp/ga_cpp/ga_bridge.so and skwp/knapsack_cpp/knap_bridge.so
```

A Gurobi licence is only needed to compute the exact baselines. The benchmarks read the stored Gurobi results from
`results/` by default (`RUN_EXACT = False`).

## Usage

All scripts must be launched **from the repository root** as modules, because the compiled libraries, models and
result files are loaded with paths relative to the root. Experiment settings (instance sizes, number of runs,
population size, iterations, variant, ...) are constants at the top of each script.

### Path Pricing Problem

```bash
python -m pp.validation    # once: reference values of the validation set -> results/validation_pp.csv
python -m pp.train         # trains the policy -> models/pp_policy.pth
python -m pp.benchmark     # test experiments -> results/pp_results.csv
```

### Stackelberg Knapsack Weight Problem

The variant is set with `VARIANT` in `skwp/train.py` and `skwp/benchmark.py`: `'k1'` for single-follower, `'mf'`
for multi-follower.

```bash
python -m skwp.validation k1   # once per variant -> results/validation_skwp_k1.csv
python -m skwp.validation mf
python -m skwp.train           # -> models/skwp_policy_<variant>.pth
python -m skwp.benchmark       # -> results/skwp_<variant>_results.csv
```

### Tables of the paper

```bash
python -m results.make_tables  # prints the LaTeX tables of the paper from results/*_results.csv
```

## Experimental setting

The main settings used in the paper (see Section 5 for details):

| | PP | SKWP ($K=1$) | SKWP ($K>1$) |
|---|---|---|---|
| Training instance sizes | $C, P \in [20, 90]$ | $M \in [20, 50]$, $K = 1$ | $M, K \in [10, 40]$ |
| Training | 3000 iterations, 32 instances per batch, 64 samples per instance | same | same |
| Validation reference | best of 5 GA runs, pop. 512, 50,000 it. | best of 5 GA runs, pop. 512, 5,000 it. | same as $K=1$ |
| GA (all variants) | pop. 128, 10,000 it. | pop. 128, 1,000 it. | pop. 128, 1,000 it. |
| Gurobi time limit | 1800 s | 1800 s | 1800 s |
| Runs per test case | 30 | 30 | 30 |

Methods compared: GTN+GA (ours), GA with random initialisation, and GA initialised with Latin Hypercube sampling and
the Sobol, Halton and Kronecker sequences. *GA-best* is the best of these five GA variants, chosen after each run.
The experiments were run on an Intel Xeon W7-2495X CPU with 128 GB RAM and an NVIDIA RTX 4090 GPU.

## Citation

```bibtex
@unpublished{gasparin2026learning,
  title   = {Learning to Initialise Genetic Algorithms in Bilevel Optimisation: A Reinforcement Learning Approach},
  author  = {Gasparin, Andrea and Camerota Verd{\`u}, Federico Julian and Castelli, Lorenzo},
  year    = {2026},
  note    = {Manuscript}
}
```

## Contact

Andrea Gasparin (andrea.gasparin@dia.units.it), Department of Engineering and Architecture, University of Trieste.
