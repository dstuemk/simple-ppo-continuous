# Continuous PPO with Tensorflow 2.0

A minimalistic implementation of OpenAI's proximal policy optimization algorithm. It learns to swing up a pendulum (from OpenAI Gym). There is much room for performance improvement, so far computation happens on only 1 CPU/GPU even if more ressources are available.

Usage:
* To train the agent run ```python main.py train```
* To run an episode (after training) run ```python main.py enjoy```

## Generalized Advantage Estimation

The advantage value is calculated using the Generalized Advantage Estimation (GAE) Method. When reading through the source code one might wonder about the usge of ```LinearOperatorToeplitz```. This operator lets us calculate the GAE values with a simple Matrix-Vector multiplication:

![Matrix-Vector multiplication](https://latex.codecogs.com/svg.image?%5Cbegin%7Bpmatrix%7D1%20&%20%5Cgamma%5Clambda%20&%20(%5Cgamma%5Clambda)%5E2%20&%20%5Chdots%20%5C%5C%20&%201%20&%20%5Cgamma%5Clambda%20&%20%20%5C%5C%20&%20%20&%201%20&%20%5Cddots%20%5C%5C%20&%20%20&%20%20&%20%5Cddots%20%5C%5C%5Cend%7Bpmatrix%7D%5Ccdot%5Cbegin%7Bpmatrix%7D%5Cdelta_1%20%5C%5C%5Cdelta_2%20%5C%5C%5Cdelta_3%20%5C%5C%5Cvdots%5Cend%7Bpmatrix%7D=%20%5Cbegin%7Bpmatrix%7D%5Chat%7BA%7D_1%20%5C%5C%5Chat%7BA%7D_2%20%5C%5C%5Chat%7BA%7D_3%20%5C%5C%5Cvdots%5Cend%7Bpmatrix%7D%20)

The ![delta](https://latex.codecogs.com/svg.image?%5Cdelta) terms are the temporal difference errors (TD-errors):

![td-error](https://latex.codecogs.com/svg.image?%5Cdelta_t%20=%20r_t%20&plus;%20%5Cgamma%20V(s_%7Bt&plus;1%7D)%20-%20V(s_t))