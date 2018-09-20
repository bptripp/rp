# Gradient descent to find random activity pattern with given representational
# similarity matrix. This works better than taking steps in null space of similarity
# matrix. 

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_random_stimuli(n_stim, dim):
    return -1 + 2 * np.random.rand(n_stim, dim)


def get_gaussian_population_response(stim, n_neurons, dim, sigma):
    centres = -1 + 2 * np.random.rand(n_neurons, dim)
    responses = np.zeros((stim.shape[0], n_neurons))
    for i in range(n_stim):
        for j in range(n_neurons):
            dist = np.linalg.norm(stim[i, :] - centres[j, :])
            responses[i, j] = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return responses


def get_tf_corrcoef(tf_responses):
    deviations = tf_responses - tf.expand_dims(tf.reduce_mean(tf_responses, axis=1), axis=1)
    numerator = tf.matmul(deviations, tf.transpose(deviations))
    d = tf.sqrt(tf.diag_part(numerator))
    return tf.div(numerator, tf.tensordot(d, d, axes=0))


def test_get_rf_corrcoef():
    responses = np.random.rand(80, 100)
    tf_corrcoef = get_tf_corrcoef(responses)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        np.testing.assert_allclose(sess.run(tf_corrcoef), np.corrcoef(responses))


def get_population_response_with_similarity(n_neurons, similarity):
    random_responses = np.random.rand(similarity.shape[0], n_neurons)

    tf_random_responses = tf.get_variable('random_responses', initializer=tf.constant(random_responses))
    tf_target_similarity = tf.constant(similarity)
    tf_actual_similarity = get_tf_corrcoef(tf_random_responses)

    cost = tf.reduce_mean(tf.pow(tf_actual_similarity - tf_target_similarity, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=.01)
    clip_op = tf.assign(tf_random_responses, tf.clip_by_value(tf_random_responses, 0, 1000000))
    opt_op = optimizer.minimize(cost, var_list=tf_random_responses)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print(sess.run(cost))
        for i in range(40):
            for j in range(25):
                opt_op.run()
                sess.run(clip_op)
            print(sess.run(cost))

        result = sess.run(tf_random_responses)

    return result


if __name__ == '__main__':
    n_neurons = 100
    n_stim = 80
    sigma = .5
    dim = 2

    # test_get_rf_corrcoef()

    stim = get_random_stimuli(n_stim, dim)
    responses = get_gaussian_population_response(stim, n_neurons, dim, sigma)
    similarity = np.corrcoef(responses)
    perturbed = get_population_response_with_similarity(n_neurons, similarity)

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
