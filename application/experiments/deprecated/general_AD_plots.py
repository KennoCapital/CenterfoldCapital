import numpy as np
import matplotlib.pyplot as plt
from application.engine.mcBase import mcSim, RNG
from application.engine.products import Caplet
from application.engine.vasicek import Vasicek
import torch
from application.utils.torch_utils import max0
from tqdm.contrib import itertools


def digital_call_payoff(S, K, P):
    return np.where(S > K, P, 0)

def call_spread_payoff(S, K, epsilon, P):
    long_call = np.maximum(S - (K - 0.5 * epsilon), 0)
    short_call = np.maximum(S - (K + 0.5 * epsilon), 0)
    return (long_call - short_call) * 1/0.01

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

def digital_approximation(S, a, b):
    return sigmoid(S, a, b)

def digital_call_option_payoff(S, strikes, epsilon, P):
    # Initialize an array to store payoffs for different strikes
    payoffs = np.zeros((len(strikes), len(S)))

    # Loop through each strike
    for i, K in enumerate(strikes):
        payoff = np.where(S > K, P, 0)
        payoffs[i, :] = payoff

    return payoffs

# Parameters
S = np.linspace(0.064, 0.104, 1000)  # underlying asset prices
K = 0.084  # strike price for digital call
P = 1  # payout for digital call

# call spread width
epsilon = 0.01

# Barrier decomposing
# Generate evenly spaced strikes in the interval (K-\varepsilon/2, K+\varepsilon/2)
num_strikes = 5  # Adjust the number of strikes as needed
K_values = np.linspace(K - 0.5 * epsilon, K + 0.5 * epsilon, num_strikes)

# for sigmoid fct
a = 250  # scaling factor
b = K  # shifting factor


## FIGURE 1 -------------------------------------
# show digital is call spread
# Payoff of Digital Call Option
digital_call_payoff_curve = digital_call_payoff(S, K, P)
# Payoff of Call Spread Replication
call_spread_payoff_curve = call_spread_payoff(S, K, epsilon, P)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(S, digital_call_payoff_curve, label='Digital Call Option Payoff', color='black')
ax.plot(S, call_spread_payoff_curve,linestyle='--', color='orange', label='Call Spread Replication Payoff')
ax.text(K - 0.5 * epsilon, max(max(digital_call_payoff_curve), max(call_spread_payoff_curve)) + 0.1,
        r'$K - \frac{\varepsilon}{2}$', color='gray', ha='center', va='center')
ax.text(K + 0.5 * epsilon, max(max(digital_call_payoff_curve), max(call_spread_payoff_curve)) + 0.1,
        r'$K + \frac{\varepsilon}{2}$', color='gray', ha='center', va='center')
ax.set_xlabel(r'$X(T_0)$')
ax.set_ylabel('Payoff')
ax.legend()
plt.show()


## FIGURE 2 -------------------------------------

# Payoffs of Digital Call Options with Different Strikes
digital_call_option_payoffs = digital_call_option_payoff(S, K_values, epsilon, P/num_strikes)

combined_payoff = 0
for i, K in enumerate(K_values):
    combined_payoff += digital_call_option_payoffs[i, :]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, digital_call_payoff_curve, label='KO Barrier', color='black')
plt.plot(S, combined_payoff, linestyle='--', color='orange', label='Partial KO Barrier')
plt.text(min(K_values), max(max(digital_call_payoff_curve), max(digital_call_option_payoffs[0, :])) + 0.1,
        r'$B - \frac{\varepsilon}{2}$', color='gray', ha='center', va='center')
plt.text(max(K_values), max(max(digital_call_payoff_curve), max(digital_call_option_payoffs[-1, :])) + 0.1,
        r'$B + \frac{\varepsilon}{2}$', color='gray', ha='center', va='center')
plt.xlabel(r'$X(T_0)$')
plt.legend()
plt.show()


## FIGURE 3 -------------------------------------
# Approximate Digital Call Option Payoff with Sigmoid

digital_approximation_curve = digital_approximation(S, a, b)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S, digital_call_payoff_curve, label='Digital Call Option Payoff', color='black')
plt.plot(S, digital_approximation_curve, color='C1', linestyle='--', label=r'Sigmoid $a_0$')
plt.plot(S,  digital_approximation(S, 750, b), color='darkorange', linestyle='--', label=r'Sigmoid $a_1$')
plt.plot(S,  digital_approximation(S, 2000, b), color='orange', linestyle='--', label=r'Sigmoid $a_2$')
plt.xlabel(r'$X(T_0)$')
plt.ylabel('Payoff')
plt.legend()
plt.show()


## FIGURE 4 -------------------
# Barrier path breach

seed = 1234
N = 100
measure = 'risk_neutral'

# vasicek params
a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)
notional = torch.tensor(1e6)

start = torch.tensor(1.0)
delta = torch.tensor(.25)
dTL = torch.linspace(0.0, start + delta, int(50 * (start + delta) + 1))

model = Vasicek(a, b, sigma, r0, True, False, measure)
swap_rate = torch.tensor(0.084)

rng = RNG(seed=seed, use_av=True)

prd = Caplet(
    strike=swap_rate,
    start=start,
    delta=delta,
    notional=notional
)

payoff = mcSim(prd, model, rng, N, dTL)

print(
    'MC price =', torch.mean(payoff)
)


fwds = torch.zeros( size=(len(model.timeline), N) )
for i, t in enumerate(model.timeline):
    fwds[i,:] = model.calc_fwd(model.x[i,:], start, delta)[0]


path = fwds[:,0:3]
eps = 0.001
barrier = path.max() - eps/4 #0.096 - eps/2

idx_barrier_hit = (path >= barrier).nonzero()
index_barrier_hit = idx_barrier_hit[0].item() + 1
new_path = path[:index_barrier_hit]

idx_lb_hit = (path >= barrier-eps/2).nonzero()
idx_lb_hit = idx_lb_hit[0].item()
lower_path = path[:idx_lb_hit]


fx1 = 0.01
fx2 = 0.
x1 = barrier-eps
x2 = barrier+eps
a = (fx1-fx2) / (x1-x2)

def line_function(x):
    return a * (x - x1) + fx1
x_subset = torch.linspace(barrier - eps, barrier + eps, 100)
# terminal
payoff = lambda x: max0( (barrier-x) / (barrier-x).abs() ) * max0(x-swap_rate)
x = torch.linspace(swap_rate-0.005, path.max()+0.005, 1000)

#plt.plot(model.timeline[:index_barrier_hit],new_path, linestyle='--' , color='black')
#plt.text(max(model.timeline[:index_barrier_hit]) +0.1 , barrier - eps / 2, r'$B-\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')
#plt.text(max(model.timeline[:index_barrier_hit])+ 0.1, barrier + eps / 2, r'$B+\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')


# path-dependent
plt.figure()
plt.plot(model.timeline,path, linestyle='--' )
plt.axhline(y=barrier, linestyle='-', color='black')
plt.axhline(y=barrier-eps/2, linestyle='-', color='orange')
plt.axhline(y=barrier+eps/2, linestyle='-', color='orange')
plt.axhline(y=path.max(), linestyle='--', color='C2')
plt.axhline(y=path[:,0].max(), linestyle='--', color='C0')
plt.axhline(y=path[:,1].max(), linestyle='--', color='C1')
plt.text(max(model.timeline) + 0.2 , barrier - eps / 2, r'$B-\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')
plt.text(max(model.timeline)+ 0.2, barrier + eps / 2, r'$B+\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')
plt.text(min(model.timeline)- 0.07, path.max(), r'$a=0$', color='C2', ha='right', va='center')
plt.text(min(model.timeline)- 0.07, path[:,0].max(), r'$a=0.25$', color='C0', ha='right', va='center')
plt.text(min(model.timeline)- 0.07, path[:,1].max(), r'$a=1$', color='C1', ha='right', va='center')
plt.xlabel('t')
plt.ylabel(r'$F(t,T, T+\delta)$')
plt.title('Up-and-Out Caplet with Discretely Monitored Barrier')
plt.show()


plt.figure()
plt.plot(x, payoff(x), color='black')
plt.axvline(x= barrier-eps, color='orange')
plt.axvline(x= barrier+eps, color='orange')
plt.plot(x_subset, line_function(x_subset), color='orange', linestyle='--')
plt.text(barrier - eps*1.2, min(payoff(x)) - 0.01/5.5, r'$B-\frac{\varepsilon}{2}$', color='orange', ha='center')
plt.text(barrier + eps*1.2, min(payoff(x)) - 0.01/5.5, r'$B+\frac{\varepsilon}{2}$', color='orange', ha='center')
plt.ylabel('Payoff')
plt.xlabel(r'$F(T,T+\delta)$')
plt.title('Up-and-Out Caplet with European Barrier')
plt.show()




# First subplot (path-dependent)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model.timeline, path, linestyle='--', color='black')
plt.axhline(y=barrier, linestyle='-', color='black')
plt.axhline(y=barrier - eps / 2, linestyle='-', color='orange')
plt.axhline(y=barrier + eps / 2, linestyle='-', color='orange')
plt.axhline(y=path.max(), linestyle='--', color='gray')
plt.text(max(model.timeline) + 0.2 , barrier - eps / 2, r'$B-\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')
plt.text(max(model.timeline)+ 0.2, barrier + eps / 2, r'$B+\frac{\varepsilon}{2}$', color='orange', ha='right', va='center')
plt.text(min(model.timeline)- 0.07, path.max(), r'$a=0.25$', color='gray', ha='right', va='center')
plt.xlabel('t')
plt.ylabel(r'$F(t,T, T+\delta)$')
plt.title('Up-and-Out Caplet with Discretely Monitored Barrier')

# Second subplot (European Barrier)
plt.subplot(1, 2, 2)
plt.plot(x, payoff(x), color='black')
plt.axvline(x=barrier - eps, color='orange')
plt.axvline(x=barrier + eps, color='orange')
plt.plot(x_subset, line_function(x_subset), color='orange', linestyle='--')
plt.text(barrier - eps * 1.2, min(payoff(x)) - 0.01 / 5.5, r'$B-\frac{\varepsilon}{2}$', color='orange', ha='center')
plt.text(barrier + eps * 1.2, min(payoff(x)) - 0.01 / 5.5, r'$B+\frac{\varepsilon}{2}$', color='orange', ha='center')
plt.ylabel('Payoff')
plt.xlabel(r'$F(T,T+\delta)$')
plt.title('Up-and-Out Caplet with Terminal Barrier')

# Adjust layout to prevent clipping of titles
plt.tight_layout(w_pad=1.)

# Show the plot
plt.show()
