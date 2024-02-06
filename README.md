# CenterfoldCapital

This repository accompany our master thesis *"Your Neural Network is Your Net Worth - Valuation and Hedging of Interest Rate Derivatives with Automatic Adjoint Differentiation and Differential Machine Learning"* to finalize our education as MSc. in Mathematics-Economics at University of Copenhagen. The thesis was written during fall 2023 and [presented](https://www.math.ku.dk/english/calendar/specialeforsvar/speciale-nkhmmaasvh/) in January 2024.

In this thesis, we have succesfully implemented a flexible and scalable setup for performing efficient valuation and accurate hedging of interest rate derivatives by Monte Carlo simulation instrumented with Automatic Adjoint Differentiation. Our main contribution demonstrates effective training of Differential Machine Learning models for a variety of interest rate products, including linear, European, path-dependent, and callable contracts. This is carried out within the classic Vasicek model, and the General Multi-Factor Stochastic Volatility Model by [Trolle and Schwartz](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=912447). 

Our findings reveal:
- Smoothing techniques can help produce better quality training samples.
- Differential Regression is fast and best applicable for nice payoff functions.
- Differential Neural Networks is generally more accurate and capable of fitting nastier functions.
- In a stochastic volatility market, Differential Neural Network seems (much more) stable for Greek estimations.

# What is Differential Machine Learning?
Differential Machine Learning in financial applications is a concept first mentioned by Savine and Huge in [Diff ML repository](https://github.com/differential-machine-learning). The idea is to combine classical supervised Machine Learning algorithms, where learning is based on inputs and outputs only, together with *differential labels*, what have the unique feature they are exactly the derivative of the output with respect to the input.

Say that we have a dataset with $N$ samples. We use supervised learning to learn the relationship (through a cost function) between input features $X$ (which in our work amounts to market observables), output labels $y$ (discounted payoffs) *and in turn also differential labels* $Z$.

Thus we need a dataset suitable for **differential learning** where each sample $i$ should contain:
$$X^{(i)} \in \mathbb{R}^n, \quad y^{(i)} \in \mathbb{R}^m, \quad Z^{(i)} \equiv \frac{\partial y^{(i)}}{\partial X^{(i)}} \in \mathbb{R}^{n\times m}.$$

Take this example, of a European payer swaption that has 1 year to expiry to enter into a 5Y3M swap. The underlying swap is our market observable $X$ from which we compute prices $y$ and differentials $Z$. We can thus employ differential learning to learn the relationship between the underlying, price and also risk figures (delta in this case).

The right pane is purely how training data look like (more on how to generate that below) and the left pane is the learned price and risk *curve* across the different values of the underlying.

<img width="517" alt="image" src="https://github.com/KennoCapital/CenterfoldCapital/assets/79045074/bd6447bd-0b8a-49b2-b03c-5278253c77c6">


# How we build a Monte Carlo simulation engine
The backbone of our data generation procedure is classical Monte Carlo simulations, but instrumented with Automatic Adjoint Differentiation (AAD).
It is thus essential that this is flexible and scalable to handle not only different products but also different models. The key to achieve this, is to segregate the responsibilities between what a *model* should do and provide, and what a *product* should do and provide.

**Product:** Determines payoff function and definition of market samples.

**Model:** Determines the probability distribution of the market samples.

Averaging over $N$ i.i.d. samples of realised discounted payoffs we obtain the Monte Carlo estimator:

$$V(t) = \mathbb{E}^Q\left\[\sum \frac{B(t)}{B(t_i)} CF(t_i; x_{t_i}) \| \mathcal{F}_t \right\] = \mathbb{E}^Q\left\[g(x) \| \mathcal{F}_t\right\] \approx \frac{1}{N}\sum g(x^i)$$

**High level implementation details of our engine:**
1. Product provides the timeline and definition line.
2. Model instructs Random Number Generator (RNG) how many discretizations is needed.
3. Number of paths to sample.
4. RNG samples and passes back to the model.
5. Product determines payoff for scenarios.

<img width="360" alt="image" src="https://github.com/KennoCapital/CenterfoldCapital/assets/79045074/8618c9b7-4721-4618-83ed-e557e3df4c66">
