# Gradient descent to find random activity pattern with given representational
# similarity matrix. This works better than taking steps in null space of similarity
# matrix.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_random_stimuli(n_stim, dim):
    """
    :param n_stim: Number of "stimuli" to sample (stimuli are vectors)
    :param dim: Dimension of each stimulus
    :return: Random stimulus vectors with entries between -1 and 1 (#stim by dim)
    """
    return -1 + 2 * np.random.rand(n_stim, dim)


def get_gaussian_population_response(stim, n_neurons, dim, sigma):
    """
    :param stim: Matrix of stimulus vectors (#stim by dim)
    :param n_neurons: Number of neurons in a population
    :param dim: Dimension of stimulus vectors
    :param sigma: Width of Gaussian tuning curves of dim dimensions
    :return: Responses of population of Gaussian tuning curves to given stimuli
    """
    centres = -1 + 2 * np.random.rand(n_neurons, dim)
    responses = np.zeros((stim.shape[0], n_neurons))
    for i in range(n_stim):
        for j in range(n_neurons):
            dist = np.linalg.norm(stim[i, :] - centres[j, :])
            responses[i, j] = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return responses


def get_tf_corrcoef(tf_responses):
    """
    :param tf_responses: A TensorFlow variable that holds a matrix of population responses to
        a list of stimuli
    :return: A TensorFlow variable that holds the matrix of correlations between responses to
        pairs of stimlui
    """
    deviations = tf_responses - tf.expand_dims(tf.reduce_mean(tf_responses, axis=1), axis=1)
    numerator = tf.matmul(deviations, tf.transpose(deviations))
    d = tf.sqrt(tf.diag_part(numerator))
    return tf.div(numerator, tf.tensordot(d, d, axes=0))


def test_get_tf_corrcoef():
    responses = np.random.rand(80, 100)
    tf_corrcoef = get_tf_corrcoef(responses)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        np.testing.assert_allclose(sess.run(tf_corrcoef), np.corrcoef(responses))


def get_population_response_with_similarity(n_neurons, similarity, initial_responses=None, iterations=10):
    """
    :param n_neurons: Number of neurons in population
    :param similarity: Correlation matrix to approximate (correlations between population responses to pairs of stimuli)
    :param initial_responses (optional): An initial guess at the population responses (initialized to a random matrix
        if not given)
    :param iterations (optional): Iterations of an outer optimization loop (several optimization steps per iteration)
    :return: Population responses that optimally approximate the similarity matrix
    """
    if initial_responses is None:
        initial_responses = np.random.rand(similarity.shape[0], n_neurons)

    tf_responses = tf.get_variable('responses', initializer=tf.constant(initial_responses))
    tf_target_similarity = tf.constant(similarity)
    tf_actual_similarity = get_tf_corrcoef(tf_responses)

    cost = tf.reduce_mean(tf.pow(tf_actual_similarity - tf_target_similarity, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=.1)
    clip_op = tf.assign(tf_responses, tf.clip_by_value(tf_responses, 0, 1000000))
    opt_op = optimizer.minimize(cost, var_list=tf_responses)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print(sess.run(cost))
        for i in range(iterations):
            for j in range(25):
                opt_op.run()
                sess.run(clip_op)
            print(sess.run(cost))

        result = sess.run(tf_responses)

    return result


def get_similarity_via_random_subsets(n_neurons, similarity, n_batch, batches):
    """
    :param n_neurons: Number of neurons in population
    :param similarity: Ccorrelation matrix to approximate (correlations between population responses to pairs of stimuli)
    :param n_batch: Size of subsets of stimuli to optimize at once
    :param batches: Number of batches to optimize
    :return: Population responses that optimally approximate the similarity matrix
    """

    #TODO: manage memory by saving and loading parts of response and similarity matrices

    n_stimuli = similarity.shape[0]
    responses = np.random.rand(n_stimuli, n_neurons)

    for i in range(batches):
        print('iteration {}'.format(i))
        subset = np.random.choice(range(n_stimuli), size=n_batch, replace=False)
        similarity_subset = similarity[subset,:][:,subset]
        responses_subset = responses[subset,:]

        tf.reset_default_graph()
        new_responses = get_population_response_with_similarity(n_neurons,
                                                                similarity_subset,
                                                                initial_responses=responses_subset,
                                                                iterations=1)
        responses[subset,:] = new_responses

    return responses


if __name__ == '__main__':
    # test_get_rf_corrcoef()

    n_neurons = 1500
    n_stim = 1000
    sigma = .5
    dim = 2

    stim = get_random_stimuli(n_stim, dim)
    responses = get_gaussian_population_response(stim, n_neurons, dim, sigma)
    similarity = np.corrcoef(responses)
    # perturbed = get_population_response_with_similarity(n_neurons, similarity)
    perturbed = get_similarity_via_random_subsets(n_neurons, similarity, 300, 50)

    plt.subplot(2,3,1)
    plt.imshow(np.corrcoef(responses), vmin=-1, vmax=1)
    plt.title('Original Similarity')
    plt.subplot(2,3,2)
    plt.imshow(np.corrcoef(perturbed), vmin=-1, vmax=1)
    plt.title('Optimized Similarity')
    plt.subplot(2,3,3)
    plt.imshow(np.corrcoef(perturbed) - np.corrcoef(responses), vmin=-1, vmax=1)
    plt.title('Difference')
    plt.subplot(2,3,4)
    plt.imshow(responses)
    plt.title('Original Responses')
    plt.subplot(2,3,5)
    plt.imshow(perturbed)
    plt.title('Optimized Responses')
    plt.subplot(2,3,6)
    plt.imshow(perturbed - responses)
    plt.title('Difference')
    plt.show()
