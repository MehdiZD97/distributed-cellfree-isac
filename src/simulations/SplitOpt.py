
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

#%% SplitOpt: Original Problem

# Problem setup
N = 4
np.random.seed(1)
alpha = (np.ones(N) / N) / 2
delta = np.array([-0.3, -0.22, 0.1, 0.11])  # Example delta values
gamma = 0.12  # Set a large gamma to test
sqrt_gamma = np.sqrt(gamma)
# Variables
rho = cp.Variable(N)
# Constraint: (sum alpha_i * sqrt(rho_i)) >= sqrt(gamma)
comm_constraint = cp.sum(cp.multiply(alpha, cp.sqrt(rho))) >= np.sqrt(gamma)
# Objective: maximize sensing utility
objective = cp.Maximize(delta @ rho)
# Problem definition
constraints = [comm_constraint, rho >= 0, rho <= 1]
problem = cp.Problem(objective, constraints)
problem.solve()
print(rho.value)
if rho.value is not None:
    plt.bar(np.arange(N), rho.value)
    plt.show()



#%% SplitOpt: Slack Variable

# Problem setup
N = 4
np.random.seed(1)
alpha = (np.ones(N) / N) / 2
delta = np.array([-0.3, -0.22, 0.1, 0.11])  # Example delta values
gamma = 0.12  # Set a large gamma to test
sqrt_gamma = np.sqrt(gamma)

# Adaptive penalty parameters
mu = 1.0
tau = 5.0  # multiplier for mu
eps = 1e-4  # acceptable slack tolerance
max_iters = 15

# Track progress
history = []

for k in range(max_iters):
    rho = cp.Variable(N)
    s = cp.Variable(nonneg=True)

    # Constraint: comm + slack >= sqrt(gamma)
    constraints = [
        cp.sum(cp.multiply(alpha, cp.sqrt(rho))) + s >= sqrt_gamma,
        rho >= 0,
        rho <= 1
    ]

    # Objective: max sensing - mu * slack
    objective = cp.Maximize(delta @ rho - mu * s)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    # Log
    history.append((mu, s.value, problem.value))

    print(f"Iter {k + 1}: mu = {mu:.2e}, slack = {s.value:.4f}, obj = {problem.value:.4f}")

    # Check if slack is negligible
    if s.value <= eps:
        print("Slack minimized. Converged.")
        break

    # Otherwise, penalize more
    mu *= tau

# Final result
rho_opt = rho.value
s_opt = s.value
print(rho_opt, s_opt)

mus, slacks, objs = zip(*history)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(mus, slacks, marker='o')
plt.xscale("log")
plt.xlabel("Penalty µ")
plt.ylabel("Slack Value")
plt.title("Slack vs Penalty µ")

plt.subplot(1, 2, 2)
plt.plot(mus, objs, marker='o')
plt.xscale("log")
plt.xlabel("Penalty µ")
plt.ylabel("Objective Value")
plt.title("Objective vs Penalty µ")

plt.tight_layout()
plt.grid(True)
plt.show()


#%% SplitOpt: Only with Feasibility Recovery

# Problem parameters
N = 4
np.random.seed(1)
alpha = (np.ones(N) / N) / 2
delta = np.array([-0.3, -0.22, 0.1, 0.11])  # Example delta values

# Feasibility recovery parameters
gamma_vals = np.linspace(0.1, 4.0, 400)  # sweep over γ
rho_solutions = []
obj_vals = []
feasible_gamma = []

# Loop over gamma values
for gamma in gamma_vals:
    rho = cp.Variable(N)
    constraint = cp.sum(cp.multiply(alpha, cp.sqrt(rho))) >= np.sqrt(gamma)
    constraints = [constraint, rho >= 0, rho <= 1]
    objective = cp.Maximize(delta @ rho)
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            rho_solutions.append(rho.value)
            obj_vals.append(problem.value)
            feasible_gamma.append(gamma)
        else:
            print(f"Infeasible at gamma = {gamma:.2f}")
            break
    except cp.error.SolverError:
        print(f"Solver failed at gamma = {gamma:.2f}")
        break

# Plotting
plt.plot(feasible_gamma, obj_vals, marker='o')
plt.xlabel("Gamma (minimum comm utility)")
plt.ylabel("Sensing Utility (Deltaᵀρ)")
plt.title("Feasibility Recovery: Sensing vs Communication Trade-off")
plt.grid(True)
plt.show()
print(rho_solutions[-1], feasible_gamma[-1])

#%% SplitOpt: Feasibility Recovery and Slack

# Problem setup
N = 4
np.random.seed(1)
alpha = (np.ones(N) / N) / 2
delta = np.array([-0.3, -0.22, 0.1, 0.11])  # Example delta values
min_rho = 0.0
max_rho = 1.0

# Gamma sweep (increasing communication requirement)
gamma_vals = np.linspace(0.01, 0.5, 20)

# Parameters
mu = 100.0  # Penalty for slack
eps = 1e-4  # Slack threshold

# Storage
sensing_utilities = []
slack_vals = []
actual_comm_utilities = []
gammas_recorded = []

# Sweep over gamma values
for gamma in gamma_vals:
    sqrt_gamma = np.sqrt(gamma)

    rho = cp.Variable(N)
    s = cp.Variable(nonneg=True)

    constraints = [
        cp.sum(cp.multiply(alpha, cp.sqrt(rho))) + s >= sqrt_gamma,
        rho >= min_rho,
        rho <= max_rho
    ]

    objective = cp.Maximize(delta @ rho - mu * s)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        sensing_utility = delta @ rho.value
        slack_used = s.value
        comm_utility = (alpha @ np.sqrt(rho.value)) ** 2

        sensing_utilities.append(sensing_utility)
        slack_vals.append(slack_used)
        actual_comm_utilities.append(comm_utility)
        gammas_recorded.append(gamma)
    else:
        print(f"Infeasible or solver error at γ = {gamma:.2f}")
        break

# Plotting results
plt.figure(figsize=(7, 5))

# plt.subplot(1, 2, 1)
plt.plot(gammas_recorded, sensing_utilities, marker='o', label="Sensing Utility")
plt.plot(gammas_recorded, actual_comm_utilities, marker='o', label="Actual Comm Utility")
plt.plot(gammas_recorded, slack_vals, marker='x', label="Slack")
plt.xlabel(r"$\gamma$ (Comm constraint)", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title(r"Utilities and Slack vs $\gamma$", fontsize=12)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

