"""
Abstract model of neural population response.
"""

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, spatial_centre, feature_centre, sigma):
        self.spatial_centre = spatial_centre
        self.feature_centre = feature_centre
        self.sigma = sigma

    def get_response(self, features):
        x = features[self.spatial_centre[0], self.spatial_centre[1], :]
        distance = np.linalg.norm(x - self.feature_centre)
        return np.exp(-distance ** 2 / (2 * self.sigma ** 2))


class Population:
    def __init__(self, n_fields, n_pixels, n_neurons, kernel_width, sigma):
        self.n_fields = n_fields
        self.n_pixels = n_pixels
        self.kernel_width = kernel_width

        self.neurons = []
        for i in range(n_neurons):
            spatial_centre = np.floor(n_pixels * np.random.rand(2)).astype(int)
            feature_centre = np.random.rand(n_fields)
            self.neurons.append(Neuron(spatial_centre, feature_centre, sigma))

    def get_features(self, stimulus):
        """
        :param stimulus: Multi-channel stimulus image
        :return: Same thing spatially filtered with centre-surround kernels.
        """
        features = np.zeros_like(stimulus)

        # gaussian_filter filters across channels too, strangely, so we'll do this channel-wise

        ratio = 3
        for i in range(stimulus.shape[2]):
            centre = scipy.ndimage.filters.gaussian_filter(stimulus[:,:,i], self.kernel_width)
            surround = scipy.ndimage.filters.gaussian_filter(stimulus[:,:,i], ratio*self.kernel_width)
            cs = centre - (1/ratio**2)*surround
            features[:,:,i] = (cs - np.min(cs)) / (np.max(cs) - np.min(cs))

        return features

    def get_responses(self, stimulus):
        features = self.get_features(stimulus)
        return [neuron.get_response(features) for neuron in self.neurons]


def make_random_stimulus(n_pixels, n_fields):
    stimulus = np.random.rand(n_pixels, n_pixels, n_fields)
    frequency_function = np.abs(1 - 2*np.linspace(0, n_pixels-1, n_pixels)/n_pixels)**2
    mask = np.outer(frequency_function, frequency_function)

    # plt.subplot(1,2,1), plt.imshow(stimulus)
    for i in range(n_fields):
        f = np.fft.fft2(stimulus[:,:,i])
        stimulus[:,:,i] = np.fft.ifft2(f * mask)

    # print(np.min(stimulus))
    # print(np.max(stimulus))
    # plt.subplot(1,2,2), plt.imshow(stimulus)
    # plt.show()

    return stimulus

if __name__ == '__main__':
    stimulus = make_random_stimulus(100, 3)

    n_fields = 3
    n_pixels = 50
    n_neurons = 100
    kernel_width = 3
    sigma = .25
    pop = Population(n_fields, n_pixels, n_neurons, kernel_width, sigma)
    features = pop.get_features(stimulus)

    responses = pop.get_responses(stimulus)
    print(responses)

    plt.plot(responses)
    plt.show()

    # print(features.shape)
    #
    # plt.subplot(1,2,1), plt.imshow(stimulus)
    # plt.subplot(1,2,2), plt.imshow(features)
    # plt.show()

