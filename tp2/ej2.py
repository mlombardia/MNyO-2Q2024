import numpy as np
import matplotlib.pyplot as plt

r = 0.1
K = 1000
A = 100


def logistic(N, t, r, K):
    return r * N * (1 - N / K)


def analytic(t, N0, r, K):
    return (K * N0 * np.exp(r * t)) / (K + N0 * (np.exp(r * t) - 1))


def logistic_allee(N, t, r, K, A):
    return r * N * (1 - N / K) * (N / A - 1)


def euler(f, N, t, dt, *args):
    return N + dt * f(N, t, *args)


def rk4(f, N, t, dt, *args):
    k1 = f(N, t, *args)
    k2 = f(N + 0.5 * k1 * dt, t + 0.5 * dt, *args)
    k3 = f(N + 0.5 * k2 * dt, t + 0.5 * dt, *args)
    k4 = f(N + k3 * dt, t + dt, *args)
    return N + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)



N0 = 100
t_start = 0
t_end = 100
dt = 0.1


t_values = np.arange(t_start, t_end, dt)
n_steps = len(t_values)


logistic_euler = np.zeros(n_steps)
logistic_sol = np.zeros(n_steps)
#logistic_sol_extra = np.zeros(n_steps) # Caso medio util para mostrar el impacto de variar r
#logistic_sol_allee_extra = np.zeros(n_steps) # Caso medio util para mostrar el impacto de variar r
logistic_sol_allee = np.zeros(n_steps)


logistic_euler[0] = N0
logistic_sol[0] = N0
#logistic_sol_extra[0] = N0
#logistic_sol_allee_extra[0] = N0
logistic_sol_allee[0] = N0

for i in range(1, n_steps):
    t = t_values[i - 1]
    logistic_euler[i] = euler(logistic, logistic_euler[i - 1], t, dt, r, K)
    logistic_sol[i] = rk4(logistic, logistic_sol[i - 1], t, dt, r, K)
    #logistic_sol_extra[i] = rk4(logistic, logistic_sol_extra[i - 1], t, dt, r-0.05, K)
    #logistic_sol_allee_extra[i] = rk4(logistic_allee, logistic_sol_extra[i - 1], t, dt, r-0.05, K, A)
    logistic_sol_allee[i] = rk4(logistic_allee, logistic_sol_allee[i - 1], t, dt, r, K, A)


#plt.plot(t_values, logistic_sol_extra, label='Logística clásica (RK4), menor r')
plt.plot(t_values, logistic_sol, label='Logística clásica (RK4)')
#plt.plot(t_values, logistic_sol_allee_extra, label='Logística con efecto Allee (RK4), menor r')
plt.plot(t_values, logistic_sol_allee, label='Logística con efecto Allee (RK4)')
plt.xlabel('Tiempo', fontsize=14)
plt.ylabel('Población N(t)', fontsize=14)
plt.legend()
plt.show()


analytic_sol = analytic(t_values, N0, r, K)

plt.plot(t_values, analytic_sol, label='Solución Analítica', linestyle='-')
plt.plot(t_values, logistic_euler, label='Solución Numérica (Euler)', linestyle=':')
plt.plot(t_values, logistic_sol, label='Solución Numérica (RK4)', linestyle='--')
plt.xlabel('Tiempo', fontsize=14)
plt.ylabel('Población N(t)', fontsize=14)
plt.legend()
plt.show()

abs_err_euler = np.abs(analytic_sol - logistic_euler)
rel_err_euler = np.sum(abs_err_euler / np.abs(analytic_sol)) / len(abs_err_euler)

abs_err = np.abs(analytic_sol - logistic_sol)
rel_err = np.sum(abs_err / np.abs(logistic_sol)) / len(abs_err)

print(rel_err_euler)
print(rel_err)

