from application.engine.mcBase import mcSim, RNG
from application.engine.products import BermudanPayerSwaption
from application.engine.vasicek import Vasicek
import torch

from sklearn.linear_model import LinearRegression
import time

torch.set_printoptions(8)
torch.set_default_dtype(torch.float64)

seed = 1234

N = 1024*500

a = torch.tensor(0.86)
b = torch.tensor(0.09)
sigma = torch.tensor(0.0148)
r0 = torch.tensor(0.08)

exerciseDates = torch.tensor([0.25, 0.5])
delta = torch.tensor(0.25)
swapLastFixingDate = torch.tensor(30.0)

t = torch.linspace(float(exerciseDates[0]),
                   float(swapLastFixingDate + delta),
                   int((swapLastFixingDate + delta - exerciseDates[0])/delta + 1))

model = Vasicek(a, b, sigma, r0, False)
swap_rate = model.calc_swap_rate(r0, t, delta)


rng = RNG(seed=seed, use_av=True)

prd = BermudanPayerSwaption(
    strike=swap_rate,
    exerciseDates=exerciseDates,
    swapLastFixingDate=swapLastFixingDate,
    delta=delta
)

print("=====================================================")
print("Bermudan swaption pricing with LSMC & Vasicek")
print("Strike = ", swap_rate)
print("Exercise dates = ", exerciseDates)
print('Accrual period = ', delta)
print("Swap last fixing date = ", swapLastFixingDate, '\n')

# start timer
start_time = time.time()
cashflows = mcSim(prd, model, rng, N)
price = torch.mean(cashflows)
print('Price computed as european=', price)

# naïve LSMC
for i in range(len(exerciseDates)-1, -1, -1):
    X = model._x[i]
    # note cashflows are shifted / given @ settlement dates
    y = cashflows[i]
    reg = LinearRegression().fit(X.reshape(-1,1), y)

    continuation_value = torch.tensor(reg.predict(X.reshape(-1,1)))
    exercise_value = y

    cashflows[i] = torch.where(continuation_value > exercise_value, continuation_value, exercise_value)

print('Naïve LSMC price =', torch.mean(cashflows.max(dim=0).values), '\n')

print('paths =', N)
print('steps =', len(t))
print("--- %s seconds ---" % (time.time() - start_time))

