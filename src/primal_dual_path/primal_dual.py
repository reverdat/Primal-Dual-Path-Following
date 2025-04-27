import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
from scipy.sparse.linalg import norm as spnorm
import sksparse.cholmod as ch

from scipy.linalg import cho_factor, cho_solve


class PrimalDualPathFollowing:
    def __init__(
        self,
        c,
        A,
        b,
        x0,
        lambda0,
        s0,
        iter_max=1000,
        epsilon_feas=1e-6,
        epsilon_opt=1e-6,
        rho=0.9999,
        min_step=1e-10,
        regularization=1e-7,
        method="predictor-corrector",  # 'predictor-corrector', 'standard'
        solver="normal",  # 'normal', 'augmented', or 'kkt'
        sigma_standard=0.1,
        verbose=True,
    ):
        """
        Initialize the Primal-Dual Path-Following Interior Point Algorithm.
        Implements Mehrotra's Predictor-Corrector and Standard Newton methods.
        Allows solving linear systems via Normal Equations, Augmented System, or Full KKT.

        Assumes the standard form LP:
        min c^T x
        s.t. Ax = b
             x >= 0

        Parameters:
        ----------
        c, A, b : numpy.ndarray
            Problem data. A can be dense or sparse.
        x0, lambda0, s0 : numpy.ndarray
            Initial strictly positive point (x0>0, s0>0).
        iter_max : int
            Maximum iterations.
        epsilon_feas, epsilon_opt : float
            Tolerances for feasibility and optimality.
        rho : float (0 < rho < 1)
            Step length reduction factor.
        min_step : float
            Minimum allowed value for x or s components.
        regularization : float
            Regularization added to diagonal of matrices for stability.
        method : str
            Algorithm variant: 'predictor-corrector' (default) or 'standard'.
        solver : str
            Linear system solver: 'normal' (default), 'augmented', or 'kkt'.
            Note: 'augmented' and 'kkt' solvers might be less numerically stable
            in this implementation compared to 'normal'.
        sigma_standard : float (0 < sigma < 1)
            Fixed centering parameter for 'standard' method (only used if method='standard').
        verbose : bool
            Print iteration details.
        """

        if not sp.issparse(A):
            A = sp.csc_matrix(A)
        elif A.format != "csc":
            A = A.tocsc()

        if not (0 < rho < 1):
            raise ValueError("Rho must be between 0 and 1.")
        if method not in ["predictor-corrector", "standard"]:
            raise ValueError("Method must be 'predictor-corrector' or 'standard'.")
        if solver not in ["normal", "augmented", "kkt"]:
            raise ValueError("Solver must be 'normal', 'augmented', or 'kkt'.")
        if not (0 < sigma_standard < 1):
            raise ValueError("sigma_standard must be between 0 and 1.")

        self.c = c.astype(float)
        self.A = A  # Keep A sparse
        self.b = b.astype(float)
        self.x = x0.astype(float)
        self.lambda_ = lambda0.astype(float)
        self.s = s0.astype(float)

        self.iter_max = int(iter_max)
        self.epsilon_feas = float(epsilon_feas)
        self.epsilon_opt = float(epsilon_opt)
        self.rho = float(rho)
        self.min_step = float(min_step)
        self.min_neg_delta = float(min_step * 1e-2)
        self.regularization = float(regularization)
        self.method = method
        self.solver = solver
        self.sigma_standard = float(sigma_standard)
        self.verbose = verbose

        self.m, self.n = self.A.shape
        self.iteration = 0
        self.status = "Not Solved"

    def _calculate_residuals_and_gap(self):
        """Calculates primal/dual residuals and duality gap."""
        rc = self.A.T @ self.lambda_ + self.s - self.c
        rb = self.A @ self.x - self.b
        gap = self.x @ self.s
        gap = max(0.0, gap)
        mu = gap / self.n if self.n > 0 else 0.0
        return rc, rb, gap, mu

    def _check_termination(self, rc_norm, rb_norm, gap):
        """Checks if termination criteria are met."""
        norm_c = max(1, np.linalg.norm(self.c))
        norm_b = max(1, np.linalg.norm(self.b))

        primal_feas_met = rb_norm / (1 + norm_b) <= self.epsilon_feas
        dual_feas_met = rc_norm / (1 + norm_c) <= self.epsilon_feas
        opt_met = gap / self.n <= self.epsilon_opt

        if primal_feas_met and dual_feas_met and opt_met:
            return True
        return False

    def _calculate_step_length(self, current_v, delta_v):
        """Calculates maximum step length for a variable v such that v + alpha*delta_v >= 0."""
        alpha_max = 1.0
        # Find indices where delta_v is significantly negative
        indices = np.where(delta_v < -self.min_neg_delta)[0]

        if indices.size > 0:
            ratios = -current_v[indices] / delta_v[indices]
            finite_ratios = ratios[np.isfinite(ratios) & (ratios >= 0)]
            if finite_ratios.size > 0:
                alpha_max = min(np.min(finite_ratios), 1.0)

        return max(0.0, alpha_max)

    def _solve_direction(self, rc, rb, rhs_comp):
        """
        Solves the Newton system for the direction (delta_x, delta_lambda, delta_s)
        using the chosen solver. Includes adaptive regularization.
        """
        n = self.n
        m = self.m
        x_safe = np.maximum(self.x, self.min_step)
        s_safe = np.maximum(self.s, self.min_step)
        mu = max(1e-12, (x_safe @ s_safe) / n)

        delta_p = 1e-8
        delta_d = 1e-12

        s_inv = 1.0 / s_safe
        x_inv = 1.0 / x_safe
        theta_diag = x_safe * s_inv

        # --- Input Sanity Checks ---
        if not np.all(np.isfinite(theta_diag)):
            raise ValueError("Non-finite Theta.")
        if not np.all(np.isfinite(rc)):
            raise ValueError("Non-finite rc.")
        if not np.all(np.isfinite(rb)):
            raise ValueError("Non-finite rb.")
        if not np.all(np.isfinite(rhs_comp)):
            raise ValueError("Non-finite rhs_comp.")

        # --- Solve based on selected method ---
        if self.solver == "normal":
            try:
                Theta = sp.diags(theta_diag, 0, format="csc")
                M = self.A @ Theta @ self.A.T
                if M.format != "csc":
                    M = M.tocsc()
                M_reg = M + sp.identity(m, format="csc") * delta_d

                rhs_term1 = s_inv * rhs_comp
                rhs_term2 = theta_diag * rc
                rhs_lambda = -rb + self.A @ rhs_term1 - self.A @ rhs_term2

                if not np.all(np.isfinite(rhs_lambda)):
                    raise ValueError("Non-finite RHS for normal equations.")

                factor = ch.cholesky(M_reg, ordering_method="amd")
                delta_lambda = factor(rhs_lambda)

                if np.any(~np.isfinite(delta_lambda)):
                    raise np.linalg.LinAlgError(
                        "CHOLMOD solver returned non-finite delta_lambda."
                    )

            except (
                ch.CholmodNotPositiveDefiniteError,
                ch.CholmodError,
                ValueError,
                RuntimeError,
                MemoryError,
            ) as e:
                raise np.linalg.LinAlgError(
                    f"Normal equations solve (CHOLMOD) failed: {e}"
                )

            try:
                delta_s = -rc - self.A.T @ delta_lambda
                delta_x = s_inv * (rhs_comp - x_safe * delta_s)
            except ValueError as e:
                raise ValueError(f"Back-substitution failed: {e}")

        elif self.solver == "augmented":
            # --- Augmented System (System 2) ---
            try:
                theta_inv_diag = s_safe * x_inv
                if not np.all(np.isfinite(theta_inv_diag)):
                    raise ValueError("Non-finite values in Theta_inv calculation.")

                neg_Theta_inv_reg = -sp.diags(theta_inv_diag, 0, format="csc")
                neg_Theta_inv_reg -= sp.identity(n) * delta_p

                neg_Reg_D = -sp.identity(m) * delta_d

                KKT_aug_upper = sp.hstack([neg_Theta_inv_reg, self.A.T], format="csc")
                KKT_aug_lower = sp.hstack([self.A, neg_Reg_D], format="csc")
                KKT_aug = sp.vstack([KKT_aug_upper, KKT_aug_lower], format="csc")

                rhs_aug1 = -rc + x_inv * rhs_comp
                rhs_aug2 = -rb
                rhs_aug = np.concatenate([rhs_aug1, rhs_aug2])

                if not np.all(np.isfinite(rhs_aug)):
                    raise ValueError("Non-finite RHS for augmented system.")

                # Solve using SciPy's default sparse solver (handles indefinite)
                sol_aug = spla.spsolve(KKT_aug, rhs_aug)

                if np.any(~np.isfinite(sol_aug)):
                    raise np.linalg.LinAlgError(
                        "Augmented solver returned non-finite values."
                    )

                delta_x = sol_aug[:n]
                delta_lambda = sol_aug[n:]
                delta_s = -rc - self.A.T @ delta_lambda

            except (np.linalg.LinAlgError, ValueError, RuntimeError, MemoryError) as e:
                raise np.linalg.LinAlgError(f"Augmented system solve failed: {e}")

        elif self.solver == "kkt":
            # --- Full KKT System (System 1) ---
            try:
                X_mat = sp.diags(x_safe, 0, format="csc")
                S_mat = sp.diags(s_safe, 0, format="csc")
                I_n = sp.identity(n, format="csc")
                Z_nn = sp.csc_matrix((n, n))
                Z_mm = sp.csc_matrix((m, m))
                Z_mn = sp.csc_matrix((m, n))
                Z_nm = sp.csc_matrix((n, m))

                KKT_r1 = sp.hstack([Z_nn, self.A.T, I_n], format="csc")
                KKT_r2 = sp.hstack([self.A, Z_mm, Z_mn], format="csc")
                KKT_r3 = sp.hstack([S_mat, Z_nm, X_mat], format="csc")

                KKT = sp.vstack([KKT_r1, KKT_r2, KKT_r3], format="csc")

                KKT[n + m :, n + m :] += sp.identity(n) * self.regularization

                rhs_kkt = np.concatenate([-rc, -rb, rhs_comp])

                if not np.all(np.isfinite(rhs_kkt)):
                    raise ValueError(
                        "Non-finite values encountered in KKT RHS calculation."
                    )

                sol_kkt = spla.spsolve(KKT, rhs_kkt)

                if np.any(~np.isfinite(sol_kkt)):
                    raise np.linalg.LinAlgError(
                        "KKT solver returned non-finite values."
                    )

                delta_x = sol_kkt[:n]
                delta_lambda = sol_kkt[n : n + m]
                delta_s = sol_kkt[n + m :]

            except (np.linalg.LinAlgError, RuntimeError, ValueError) as e:
                raise np.linalg.LinAlgError(f"Full KKT system solve failed: {e}")

        else:
            raise ValueError(f"Unknown solver type: {self.solver}")

        # --- Final checks on directions ---
        if (
            not np.all(np.isfinite(delta_x))
            or not np.all(np.isfinite(delta_lambda))
            or not np.all(np.isfinite(delta_s))
        ):
            details = []
            if not np.all(np.isfinite(delta_x)):
                details.append("delta_x")
            if not np.all(np.isfinite(delta_lambda)):
                details.append("delta_lambda")
            if not np.all(np.isfinite(delta_s)):
                details.append("delta_s")
            raise ValueError(
                f"Non-finite values encountered in calculated directions: {', '.join(details)}."
            )

        return delta_x, delta_lambda, delta_s

    def solve(self):
        """
        Solve the linear programming problem using the chosen method and solver.
        """
        if self.verbose:
            print(
                f"--- Primal-Dual IPM ({self.method}, {self.solver} solver) Start ---"
            )
            print(
                f"{'Iter':>4s} | {'Primal Obj':>12s} | {'Dual Obj':>12s} | {'Gap':>12s} | {'Feas Primal':>12s} | {'Feas Dual':>12s} | {'Mu':>12s}"
            )
            print("-" * 85)

        e = np.ones(self.n)

        for k in range(self.iter_max):
            self.iteration = k

            self.x = np.maximum(self.x, self.min_step)
            self.s = np.maximum(self.s, self.min_step)

            rc, rb, gap, mu = self._calculate_residuals_and_gap()
            rc_norm = np.linalg.norm(rc)
            rb_norm = np.linalg.norm(rb)

            if self.verbose:
                primal_obj = self.c @ self.x
                dual_obj = self.b @ self.lambda_
                print(
                    f"{k:4d} | {primal_obj: 12.4e} | {dual_obj: 12.4e} | {gap: 12.4e} | {rb_norm: 12.4e} | {rc_norm: 12.4e} | {mu: 12.4e}"
                )

            if self._check_termination(rc_norm, rb_norm, gap):
                self.status = "Optimal solution found."
                break

            # --- Heuristic stop conditions ---
            if mu < self.epsilon_opt * 1e-8:
                self.status = (
                    f"Mu ({mu:.2e}) significantly smaller than tolerance, stopping."
                )
                break
            if (
                not np.isfinite(mu)
                or not np.isfinite(self.c @ self.x)
                or not np.isfinite(gap)
            ):
                self.status = "Non-finite values encountered (mu, objective, or gap)."
                warnings.warn(
                    f"Solver stopped at iter {k} due to non-finite values. Returning last valid iterate.",
                    RuntimeWarning,
                )
                break

            # ====== Calculate Search Direction ======
            if self.method == "predictor-corrector":
                # --- Predictor Step (Affine, sigma=0) ---
                rhs_comp_aff = -self.x * self.s
                delta_x_aff, delta_lambda_aff, delta_s_aff = self._solve_direction(
                    rc, rb, rhs_comp_aff
                )

                # --- Calculate Affine Step Lengths & Adaptive Sigma ---
                alpha_p_aff = self._calculate_step_length(self.x, delta_x_aff)
                alpha_d_aff = self._calculate_step_length(self.s, delta_s_aff)

                if not (np.isfinite(alpha_p_aff) and np.isfinite(alpha_d_aff)):
                    raise ValueError("Non-finite affine step length calculated.")
                mu_aff = max(
                    0.0,
                    (self.x + alpha_p_aff * delta_x_aff)
                    @ (self.s + alpha_d_aff * delta_s_aff)
                    / self.n,
                )
                sigma = np.clip((mu_aff / mu) ** 3 if mu > 1e-12 else 0.1, 0.01, 0.99)

                # --- Corrector Step (Centering + Correction) ---
                rhs_comp_cc = (
                    -self.x * self.s + sigma * mu * e - delta_x_aff * delta_s_aff
                )
                delta_x, delta_lambda, delta_s = self._solve_direction(
                    rc, rb, rhs_comp_cc
                )

            else:  # Standard Newton method
                sigma = self.sigma_standard
                rhs_comp = -self.x * self.s + sigma * mu * e
                delta_x, delta_lambda, delta_s = self._solve_direction(rc, rb, rhs_comp)
            # ==============================================================

            # --- Calculate Final Step Lengths ---
            alpha_p = self._calculate_step_length(self.x, delta_x)
            alpha_d = self._calculate_step_length(self.s, delta_s)
            alpha_p *= self.rho
            alpha_d *= self.rho
            alpha_p = max(0.0, alpha_p)
            alpha_d = max(0.0, alpha_d)

            # --- Check for Stagnation ---
            zero_step_tol = self.min_step * 10
            if k > 5 and alpha_p < zero_step_tol and alpha_d < zero_step_tol:
                self.status = "Stagnation detected (zero step length calculated)."
                warnings.warn(
                    f"Solver stopped at iter {k} due to stagnation (zero step length). Returning last valid iterate.",
                    RuntimeWarning,
                )
                break

            step_norm_p = (
                np.linalg.norm(alpha_p * delta_x)
                if np.all(np.isfinite(alpha_p * delta_x))
                else 0
            )
            step_norm_d = (
                np.linalg.norm(alpha_d * delta_s)
                if np.all(np.isfinite(alpha_d * delta_s))
                else 0
            )
            stagnation_tol = self.epsilon_feas * 1e-1
            if k > 10 and step_norm_p < stagnation_tol and step_norm_d < stagnation_tol:
                self.status = "Stagnation detected (very small step size)."
                warnings.warn(
                    f"Solver stopped at iter {k} due to stagnation. Returning last valid iterate.",
                    RuntimeWarning,
                )
                break

            # --- Update Variables ---
            self.x += alpha_p * delta_x
            self.lambda_ += alpha_d * delta_lambda
            self.s += alpha_d * delta_s

        # --- End of Loop ---
        if k == self.iter_max - 1 and self.status == "Not Solved":
            self.status = "Maximum number of iterations reached."

        if self.verbose:
            print("-" * 85)
            print(f"--- {self.status} ---")
            if isinstance(self.x, np.ndarray) and np.all(np.isfinite(self.x)):
                print(f"Final Primal Objective: {self.c @ self.x:.6e}")
                if isinstance(self.s, np.ndarray) and np.all(np.isfinite(self.s)):
                    print(f"Final Duality Gap: {self.x @ self.s:.6e}")
                else:
                    print("Final Duality Gap: Invalid (s is non-finite)")
                print(
                    f"Final Primal Feasibility: {np.linalg.norm(self.A @ self.x - self.b):.6e}"
                )
                if (
                    isinstance(self.lambda_, np.ndarray)
                    and np.all(np.isfinite(self.lambda_))
                    and isinstance(self.s, np.ndarray)
                    and np.all(np.isfinite(self.s))
                ):
                    print(
                        f"Final Dual Feasibility: {np.linalg.norm(self.A.T @ self.lambda_ + self.s - self.c):.6e}"
                    )
                else:
                    print("Final Dual Feasibility: Invalid (lambda or s is non-finite)")
            else:
                print("Final solution is not valid (contains non-finite values).")
            print(f"Iterations: {self.iteration + 1}")

        return self.x, self.lambda_, self.s, self.status, self.iteration + 1
