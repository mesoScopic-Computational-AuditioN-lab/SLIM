# SLIM

Code for the Simultaneous Learning and Inference Model (SLIM).

## Abstract

Perception has been proposed to be not a result of merely an integration of noisy sensory input but an inferential process that combines sensory evidence with prior expectations to infer the most probable latent states of the environment. Exact Bayesian inference is intractable in the continuous, high-dimensional state spaces of the natural world. Variational schemes address this by restricting the form of the posterior, whereas sampling schemes represent the posterior with a finite set of samples. In both cases the approximation concerns the posterior rather than the state space over which it is defined. Here we place the approximation elsewhere. Rather than restricting the form of the posterior, we restrict the state space, modeling environmental states as discrete and thereby making Bayesian filtering exact. Beliefs are then unconstrained in shape and what has to be learned can be reduced to a transition matrix updated online by a local rule. The cost is that the states, rather than the distribution, must be specified in advance. The Simultaneous Learning and Inference Model (SLIM) combines this exact filtering with a gated Hebbian rule that learns transitions from inferred rather than observed states. In simulations SLIM recovers environmental dynamics under sensory noise and adapts when those dynamics change, with error growing only once the sensor becomes uninformative. A hierarchical instantiation reproduces local and global prediction error effects. Applied to two auditory decision-making experiments under noise, SLIM reproduces behavioral signatures of expectation and supplies trial-level measures of expectation and surprise. Exact inference over a small discrete state space with a single local Hebbian rule is therefore sufficient to account for these phenomena, without an explicit optimization objective and without a parametric approximation to the posterior.
## Features

The SLIM model can
* learn state transitions within noisy, stochastic, and non-stationary environments,
* capture regularities that span different timescales,
* explain empirical results related to predictive processing,
* guide hypotheses about predictive processing.



## Installation

You can install the SLIM model using pip. After cloning/downloading the package, navigate to `/path/to/SLIM/` and run the following:

```bash
pip install .
```

## Quick Start (Tutorial)

This short tutorial shows a common use case for SLIM.

### 1. Import the package

```python
from slim.model import SLIM

import numpy as np
import matplotlib.pyplot as plt
```

### 2. Create a sequence of states

```python
# create 30 repititions of states 1-0-2 followed by 2-1-0
sequence = [1, 0, 2] * 30 + [2, 1, 0]
```

### 3. Build model

```python
model = SLIM(dim=3, sensor=np.eye(3))  # here we assume ideal observation.
```

### 3. Train model

```python
model.fit(sequence, learning_rate=0.1)
```

or equivalently

```python
for x in sequence:
    y = model.observe(x)
    model.step(y, learning_rate=0.1)

```

The latter gives you more flexibility, e.g. for adjusting the sensor at each iteration.

### 4. Analyse results

Here we show some useful plots based on what the model has been trained with:

**State Prediction Error**

```python
plt.figure(figsize=(20,1))
plt.plot(np.array(model.state_prediction_errors))
plt.savefig('./docs/state_prediction_errors.png')
```
![Package workflow](docs/state_prediction_errors.png)

Note that state prediction errors are not directly used in the model, Instead we use a more informative signal which is an error of the transitions, i.e. *inferred - expected* transition. We call this signal a context weighted error potential. It can be read using `model.context_weighted_error_potential`. Note that the norm of this signal is equivalent to the `state_prediction_error`.

**Learned Internal Representation**

```python
from slim import viz
viz.create_digraph(model, states=['A','B','C'], axis=plt.subplot())
plt.savefig('./docs/learned_model.png')
```
![Package workflow](docs/learned_model.png)


**Beliefs (Posteriors)**

```python
plt.figure(figsize=(20,10))
plt.imshow(model.bel.T)
plt.savefig('./docs/model_beliefs.png')
```
![Package workflow](docs/model_beliefs.png)

**State Predictions** 
```python
plt.figure(figsize=(20,10))
plt.imshow(np.array(model.x_hat_hist).T)
plt.savefig('./docs/model_predictions.png')
```
![Package workflow](docs/model_predictions.png)

## License

All rights reserved.
