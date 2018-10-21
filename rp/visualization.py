"""
Plots examples of 1D Gaussian tuning curves, others that are linear combinations of
these, and others that are optimized to share the same correlation matrix. In one
case linear recontruction is good, and in the other case correlations match well.
In both cases, tuning curve statistics are different.
"""

import numpy as np
import matplotlib.pyplot as plt
from rp.rsa import get_random_stimuli, get_gaussian_population_response, get_population_response_with_similarity

np.random.seed(2)

dim = 1
n_stim = 100
n_neurons = 5

# stimuli = get_random_stimuli(n_stim, dim)
# stimuli = list(stimuli)
# stimuli.sort()
# stimuli = np.array(stimuli)
stimuli = np.linspace(-1, 1, n_stim)
stimuli = stimuli[:,None]
centres = np.linspace(-1, 1, n_neurons)
centres = centres[:,None]
responses = get_gaussian_population_response(stimuli, n_neurons, dim, .2, centres=centres)


similarity = np.corrcoef(responses)
# initial_responses = np.dot(responses, np.random.rand(10,10))
# initial_responses = np.dot(responses, .5*(np.random.rand(10,10)<.3))
initial_responses = get_gaussian_population_response(stimuli, n_neurons, dim, .05, centres=centres)
initial_responses = np.zeros_like(responses)
for i in range(len(centres)):
    initial_responses[:,i] = .5 + .5*np.sin(2*np.pi*stimuli.flatten() + 2*np.pi*np.random.rand(1))
# plt.plot(stimuli, initial_responses)
# plt.show()

# initial_responses = None
perturbed_responses = get_population_response_with_similarity(n_neurons, similarity, initial_responses=initial_responses)
perturbed_similarity = np.corrcoef(perturbed_responses)
perturbed_weights, residuals, rank, s = np.linalg.lstsq(perturbed_responses, responses)

mapped_responses = np.dot(responses, np.random.rand(n_neurons,n_neurons))
mapped_similarity = np.corrcoef(mapped_responses)
mapped_weights, residuals, rank, s = np.linalg.lstsq(mapped_responses, responses)


# plt.axes((0.125,0.653529,0.352273,0.226471))
plt.subplot(3,3,1)
plt.title('Tuning Curves')
plt.plot(stimuli, responses)
plt.xticks([])
plt.yticks([])
plt.ylabel('Original', fontsize=12)
plt.subplot(3,3,2)
plt.title('Linear Reconstructions')
plt.axis('off')
# plt.gca()

plt.subplot(3,3,4)
plt.ylabel('RSM Fit', fontsize=12)
plt.plot(stimuli, perturbed_responses)
plt.xticks([])
plt.yticks([])
plt.subplot(3,3,5)
plt.plot(stimuli, np.dot(perturbed_responses, perturbed_weights))
plt.xticks([])
plt.yticks([])

plt.subplot(3,3,7)
plt.ylabel('Weighted Sum', fontsize=12)
plt.plot(stimuli, mapped_responses)
plt.yticks([])
plt.xlabel('Stimulus', fontsize=12)
plt.subplot(3,3,8)
plt.plot(stimuli, np.dot(mapped_responses, mapped_weights))
plt.yticks([])
plt.xlabel('Stimulus', fontsize=12)

plt.subplot(3,3,3)
plt.title('RSM')
plt.imshow(similarity, vmin=-1, vmax=1), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6)
plt.imshow(perturbed_similarity, vmin=-1, vmax=1), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9)
plt.imshow(mapped_similarity, vmin=-1, vmax=1), plt.xticks([]), plt.yticks([])
# plt.imshow(perturbed_similarity - similarity, vmin=-1, vmax=1)
plt.show()

