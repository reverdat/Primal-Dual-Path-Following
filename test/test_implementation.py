import pytest  # Import pytest
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import warnings
import time
import os
from amplpy import AMPL, Environment, AMPLException

from src.primal_dual_path.primal_dual import PrimalDualPathFollowing

NETLIB_DIR = os.path.join(
    "/Users/arnauperezreverte/Documents/MESIO/large_scale_optimization/Primal-Dual-Path/",
    "data",
    "mat",
)
AMPL_MODEL_PATH = os.path.join(
    "/Users/arnauperezreverte/Documents/MESIO/large_scale_optimization/Primal-Dual-Path/",
    "src",
    "ampl",
    "linear.mod",
)


def _load_netlib(problem_name, data_dir=NETLIB_DIR):
    """Loads a Netlib problem from a .mat file."""
    mat_file_path = os.path.join(data_dir, f"{problem_name}.mat")
    if not os.path.exists(mat_file_path):
        alt_path = os.path.join(os.path.dirname(__file__), problem_name + ".mat")
        if os.path.exists(alt_path):
            mat_file_path = alt_path
        else:
            pytest.skip(f"Netlib file not found: {mat_file_path} or {alt_path}")

    try:
        mat_file = sio.loadmat(mat_file_path)
        mat_data = mat_file["Problem"]
        bounds_data = mat_data[0, 0][5]

        A = mat_data[0, 0][2]
        b = mat_data[0, 0][3].reshape((A.shape[0],))
        c = bounds_data[0, 0]["c"].reshape((1, -1))[0]
        if sp.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = np.asarray(A)

        m, n = A_dense.shape
        if len(c) != n or len(b) != m:
            raise ValueError(f"Dim mismatch in {problem_name}")
        print(f"\nLoaded Netlib '{problem_name}': m={m}, n={n}")
        return A_dense, b, c
    except Exception as e:
        raise RuntimeError(f"Error loading {problem_name}: {e}")


def verify_with_ampl(c, A, b, ampl_model_path=AMPL_MODEL_PATH):
    """Solves the LP using AMPL/CPLEX and returns solution, objective, and status."""
    if not os.path.exists(ampl_model_path):
        fallback_path = os.path.join(os.path.dirname(__file__), "linear.mod")
        if os.path.exists(fallback_path):
            ampl_model_path = fallback_path
        else:
            pytest.skip(
                f"AMPL model file not found: {ampl_model_path} or {fallback_path}"
            )

    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A)

    ampl = AMPL(Environment(""))
    x_ampl = None
    obj_ampl = np.nan
    solve_result = "unknown"
    try:
        ampl.read(ampl_model_path)
        ampl.setOption("solver", "cplex")
        ampl.setOption("presolve", 0)
        n_vars, n_constr = len(c), len(b)
        vars_set = [str(i + 1) for i in range(n_vars)]
        constr_set = [str(i + 1) for i in range(n_constr)]
        ampl.set["VARS"] = vars_set
        ampl.set["CONSTR"] = constr_set
        ampl.param["c"] = {str(i + 1): float(c[i]) for i in range(n_vars)}
        ampl.param["b"] = {str(i + 1): float(b[i]) for i in range(n_constr)}
        A_dict = {
            (str(i + 1), str(j + 1)): float(A_dense[i, j])
            for i in range(n_constr)
            for j in range(n_vars)
            if abs(A_dense[i, j]) > 1e-15
        }
        ampl.param["A"] = A_dict
        print("Solving with AMPL/CPLEX...")
        solve_start = time.time()
        try:
            ampl.solve()
            solve_result = ampl.get_value("solve_result")
            if not solve_result:
                solve_result = "error"
        except AMPLException as ampl_e:
            print(f"AMPL solve failed: {ampl_e}")
            try:
                solve_result = ampl.get_value("solve_result")
            except Exception:
                solve_result = "error"
            if not solve_result:
                solve_result = "error"
        solve_time = time.time() - solve_start
        print(f"AMPL/CPLEX Time: {solve_time:.4f}s")
        print(f"AMPL solve result: {solve_result}")
        if "solved" in solve_result:
            try:
                obj_ampl = ampl.getObjective("objective").value()
            except Exception:
                print("Warn: Could not get AMPL objective.")
            try:
                x_ampl = np.array(
                    [ampl.get_value(f"x['{j+1}']") for j in range(n_vars)]
                )
            except Exception:
                print("Warn: Could not get AMPL solution.")
        elif "infeasible" in solve_result or "unbounded" in solve_result:
            print(f"AMPL status: {solve_result}")
        else:
            print(f"AMPL status '{solve_result}' not optimal.")
    except Exception as e:
        print(f"Error during AMPL setup/exec: {e}")
        solve_result = "error"
    finally:
        ampl.close()
    return x_ampl, obj_ampl, solve_result


# ==============================================================================
# Pytest Parameterization
# ==============================================================================

NETLIB_PROBLEMS_TO_TEST = [
    "lp_afiro",
    "lp_israel",
    "lpi_galenet",
    "lp_stocfor1",
    "lp_sierra",
    "lp_pilot87",
    "lp_stair",
]
IPM_METHODS_TO_TEST = ["standard", "predictor-corrector"]
IPM_SOLVERS_TO_TEST = ["kkt"]

test_params = [
    (problem, method, solver)
    for problem in NETLIB_PROBLEMS_TO_TEST
    for method in IPM_METHODS_TO_TEST
    for solver in IPM_SOLVERS_TO_TEST
]


# --- Pytest Test Function ---
@pytest.mark.parametrize("problem_name, ipm_method, ipm_solver", test_params)
def test_ipm_vs_ampl(problem_name, ipm_method, ipm_solver):
    """
    Tests the custom IPM solver against AMPL/CPLEX for a given Netlib problem,
    parameterizing over problem, method, and solver type.
    """
    print(
        f"\nTesting Problem: {problem_name}, Method: {ipm_method}, Solver: {ipm_solver}"
    )

    try:
        # Load A as sparse
        A_sparse, b, c = _load_netlib(problem_name)
        m, n = A_sparse.shape
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        pytest.fail(f"Failed to load problem {problem_name}: {e}")

    x0 = np.ones(n) * 100
    s0 = np.ones(n) * 100
    lambda0 = np.zeros(m)
    ipm_params = {
        "method": ipm_method,
        "solver": ipm_solver,
        "rho": 0.9999,
        "iter_max": 500,
        "regularization": 1e-7,
        "sigma_standard": 0.1,
        "verbose": True,
    }

    print("Running Custom IPM...")
    ipm_start_time = time.time()
    solver_inst = PrimalDualPathFollowing(
        c.copy(),
        A_sparse.copy(),
        b.copy(),
        x0.copy(),
        lambda0.copy(),
        s0.copy(),
        **ipm_params,
    )
    x_ipm, _, _, status_ipm, iter_ipm = solver_inst.solve()
    ipm_time = time.time() - ipm_start_time
    print(f"IPM Status: {status_ipm} ({iter_ipm} iterations), Time: {ipm_time:.4f}s")

    try:
        x_ampl, obj_ampl, status_ampl = verify_with_ampl(c, A_sparse, b)
    except Exception as e:
        pytest.fail(f"AMPL verification failed unexpectedly for {problem_name}: {e}")

    # --- Assertions ---
    is_known_infeasible = problem_name.split("_")[0] == "-"

    if "infeasible" in status_ampl or status_ampl == "error" or is_known_infeasible:
        print(
            f"AMPL reported status: {status_ampl} (or known infeasible). Checking if IPM avoided claiming optimality."
        )
        if is_known_infeasible and "solved" in status_ampl:
            print(
                f"Warning: CPLEX reported 'solved' for known infeasible '{problem_name}'. Allowing non-optimal IPM status."
            )
            assert (
                status_ipm != "Optimal solution found."
            ), f"IPM claimed optimality for known infeasible problem '{problem_name}'."
        elif status_ampl != "solved":  # For other infeasible/error cases
            assert (
                status_ipm != "Optimal solution found."
            ), f"IPM claimed optimality for problem '{problem_name}' which AMPL found '{status_ampl}'."
        print(
            f"Test PASSED for {problem_name} ({ipm_method}, {ipm_solver}) (Correctly handled non-optimal status: {status_ampl})"
        )

    elif "solved" in status_ampl:
        print("AMPL reported optimal solution. Comparing results.")
        assert isinstance(x_ipm, np.ndarray), "IPM did not return a numpy array for x."
        assert np.all(np.isfinite(x_ipm)), "IPM solution contains non-finite values."
        assert (
            x_ampl is not None
        ), "AMPL reported 'solved' but did not return a solution vector."
        assert np.all(np.isfinite(x_ampl)), "AMPL solution contains non-finite values."
        assert not np.isnan(obj_ampl), "AMPL objective is NaN despite 'solved' status."

        is_ipm_optimal = status_ipm == "Optimal solution found."
        is_ipm_potentially_optimal = is_ipm_optimal
        if status_ipm == "Maximum iterations reached.":
            rc_final, rb_final, gap_final, mu_final = (
                solver_inst._calculate_residuals_and_gap()
            )
            if solver_inst._check_termination(
                np.linalg.norm(rc_final), np.linalg.norm(rb_final), gap_final
            ):
                is_ipm_potentially_optimal = True
                print("IPM reached max iterations but met termination criteria.")

        if not is_ipm_potentially_optimal:
            warnings.warn(
                f"IPM did not converge optimally for {problem_name} ({ipm_method}, {ipm_solver}). Status: {status_ipm}"
            )

        if is_ipm_potentially_optimal:
            obj_ipm = c @ x_ipm
            print(f"IPM Objective: {obj_ipm:.8e}")
            print(f"AMPL Objective: {obj_ampl:.8e}")
            assert obj_ipm == pytest.approx(
                obj_ampl, rel=5e-4
            ), f"Objective values differ significantly."
            print(
                f"Test PASSED for {problem_name} ({ipm_method}, {ipm_solver}) (Optimal solution match)"
            )
        else:
            print(
                f"Skipping objective comparison for {problem_name} ({ipm_method}, {ipm_solver}) as IPM did not converge optimally."
            )
            pytest.fail(
                f"IPM did not converge optimally for {problem_name} ({ipm_method}, {ipm_solver}) when AMPL did. Status: {status_ipm}"
            )

    elif "unbounded" in status_ampl:
        print(
            f"AMPL reported status: {status_ampl}. Checking if IPM avoided claiming optimality."
        )
        assert (
            status_ipm != "Optimal solution found."
        ), f"IPM claimed optimality for a problem AMPL found '{status_ampl}'."
        print(
            f"Test PASSED for {problem_name} ({ipm_method}, {ipm_solver}) (Correctly handled non-optimal status: {status_ampl})"
        )

    else:
        pytest.fail(f"AMPL reported an unexpected status: {status_ampl}")
