import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

import numpy as np

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_spectrum_allocation_qp(R: np.ndarray, P1: float = 300.0):
    U, F = R.shape
    qp = QuadraticProgram(name="6g_spectrum_allocation")

    for u in range(U):
        for f in range(F):
            qp.binary_var(name=f"x_{u}_{f}")

    linear = {f"x_{u}_{f}": float(R[u, f]) for u in range(U) for f in range(F)}
    qp.maximize(linear=linear)

    for f in range(F):
        coeffs = {f"x_{u}_{f}": 1 for u in range(U)}
        qp.linear_constraint(linear=coeffs, sense="<=", rhs=1, name=f"exclusive_band_{f}")

    qubo = QuadraticProgramToQubo(penalty=P1).convert(qp)
    return qubo


def get_sampler():
    try:
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler()
    except Exception:
        pass

    try:
        from qiskit.primitives import SamplerV2
        return SamplerV2()
    except Exception:
        pass

    try:
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        return AerSamplerV2()
    except Exception as e:
        raise RuntimeError("No working V2 sampler found") from e


def solve_qubo_with_qaoa(qubo, reps: int = 1, maxiter: int = 50):
    sampler = get_sampler()
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    return MinimumEigenOptimizer(qaoa).solve(qubo)


def decode_solution(result, U: int, F: int) -> np.ndarray:
    x = np.zeros((U, F), dtype=int)
    d = result.variables_dict
    for u in range(U):
        for f in range(F):
            x[u, f] = int(round(d.get(f"x_{u}_{f}", 0)))
    return x


def pretty_print_allocation(x: np.ndarray, R: np.ndarray):
    print("x=")
    print(x)
    print("utility=", float(np.sum(R * x)))
    print("per_band_sum=", np.sum(x, axis=0))


def main():
    U, F = 4, 3
    np.random.seed(1)
    R = np.round(np.random.uniform(1.0, 10.0, size=(U, F)), 3)

    print("R=")
    print(R)

    qubo = build_spectrum_allocation_qp(R, P1=300.0)
    result = solve_qubo_with_qaoa(qubo, reps=1, maxiter=50)

    x = decode_solution(result, U, F)
    pretty_print_allocation(x, R)


if __name__ == "__main__":
    main()
