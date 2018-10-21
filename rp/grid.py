import numpy as np
import matplotlib.pyplot as plt
from rp.rsa import get_random_stimuli, get_gaussian_population_response, get_population_response_with_similarity
from rp.population import Population, make_random_stimulus

def get_errors(n_fields, n_pixels, n_neurons, kernel_width, sigma, n_stim):
    pop = Population(n_fields, n_pixels, n_neurons, kernel_width, sigma)

    responses = np.zeros((n_stim, n_neurons))
    for i in range(n_stim):
        stimulus = make_random_stimulus(n_pixels, n_fields)
        responses[i,:] = pop.get_responses(stimulus)

    similarity = np.corrcoef(responses)

    perturbed_responses = get_population_response_with_similarity(n_neurons, similarity)
    perturbed_similarity = np.corrcoef(perturbed_responses)

    # hold out 20% of stimuli for linear fit
    n_train = int(.8*n_stim)
    train_r = responses[:n_train,:]
    train_pr = perturbed_responses[:n_train,:]
    perturbed_weights, residuals, rank, s = np.linalg.lstsq(train_pr, train_r)
    approx = np.dot(perturbed_responses, perturbed_weights)

    similarity_difference = (similarity - perturbed_similarity)
    np.fill_diagonal(similarity_difference, np.nan)
    similarity_error_sd = np.nanstd(similarity_difference)

    similarity_copy = similarity.copy()
    np.fill_diagonal(similarity_copy, np.nan)
    similarity_sd = np.nanstd(similarity_copy)
    similarity_error = similarity_error_sd / similarity_sd


    # similarity_error = np.mean((similarity - perturbed_similarity).flatten()**2)**.5
    # train_error = np.mean((approx[:n_train,:] - responses[:n_train,:]).flatten()**2)**.5
    # test_error = np.mean((approx[n_train:,:] - responses[n_train:,:]).flatten()**2)**.5
    train_error_sd = np.std(approx[:n_train,:] - responses[:n_train,:])
    test_error_sd = np.std(approx[n_train:, :] - responses[n_train:, :])

    train_error = train_error_sd / np.std(responses)
    test_error = test_error_sd / np.std(responses)

    # plt.subplot(1,2,1), plt.imshow(similarity)
    # plt.subplot(1,2,2), plt.imshow(perturbed_similarity)
    # plt.show()

    # return similarity_error, test_error, np.std(responses), train_error
    return similarity_error, train_error, test_error


# np.random.seed(2)

reps = 2
# n_fields = (1,2,3)
n_fields = 3
n_pixels = 50
# n_neurons = 100
n_neurons = [10, 100, 1000]
kernel_width = 10
sigma = .5
n_stim = 100

cases = n_neurons

similarity_errors = np.zeros((reps, len(cases)))
test_errors = np.zeros((reps, len(cases)))
# response_sd = np.zeros((reps, len(cases)))
train_errors = np.zeros((reps, len(cases)))

for i in range(len(cases)):
    for j in range(reps):
        # similarity_errors[j,i], test_errors[j, i], response_sd[j, i], train_errors[j, i] \
        #     = get_errors(n_fields, n_pixels, n_neurons[i], kernel_width, sigma, n_stim)
        similarity_errors[j,i], train_errors[j, i], test_errors[j, i] \
            = get_errors(n_fields, n_pixels, n_neurons[i], kernel_width, sigma, n_stim)

plt.plot(cases, np.mean(similarity_errors, axis=0))
# plt.plot(cases, np.mean(response_sd, axis=0))
plt.plot(cases, np.mean(train_errors, axis=0))
plt.plot(cases, np.mean(test_errors, axis=0))
# plt.legend(('similarity error', 'test error', 'response SD', 'train error'))
plt.legend(('similarity error', 'train error', 'test error'))
plt.show()

# print('Similarity error: {}  Approx error: {}'.format(similarity_error, approx_error))
