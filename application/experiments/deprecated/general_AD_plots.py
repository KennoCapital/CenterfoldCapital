import numpy as np
import matplotlib.pyplot as plt

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
plt.plot(S, digital_approximation_curve, color='darkorange', linestyle='--', label=r'Sigmoid $a_0$')
plt.plot(S,  digital_approximation(S, 750, b), color='orange', linestyle='--', label=r'Sigmoid $a_1$')
plt.plot(S,  digital_approximation(S, 2000, b), color='bisque', linestyle='--', label=r'Sigmoid $a_2$')
plt.xlabel(r'$X(T_0)$')
plt.ylabel('Payoff')
plt.legend()
plt.show()



