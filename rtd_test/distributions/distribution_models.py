import numpy as np
import theano
import theano.tensor as T
import lasagne

from ..utils import tolist
from .distribution_samples import (
    DeterministicSample,
    BernoulliSample,
    CategoricalSample,
    GaussianSample,
    GaussianConstantVarSample,
    LaplaceSample,
    KumaraswamySample,
    GammaSample,
    BetaSample,
    DirichletSample,
)


class Distribution(object):
    """
    Paramaters
    ----------
    mean_network : lasagne.layers.Layer
       The network whose outputs express the paramater of this distribution.

    given : list
       This contains instances of lasagne.layers.InputLayer, which mean the
       conditioning variables.
       e.g. if given = [x,y], then the corresponding log-likehood is
            log p(*|x,y)
    """

    def __init__(self, distribution, mean_network, given, seed=1,
                 set_log_likelihood=True):
        self.distribution = distribution
        self.mean_network = mean_network
        self.given = given
        self.inputs = [x.input_var for x in given]
        _output_shape = self.get_output_shape()
        self.output = T.TensorType('float32', (False,) * len(_output_shape))()

        self.set_log_likelihood = set_log_likelihood
        self.set_seed(seed=seed)

    def set_seed(self, seed=1):
        # Need to call set_seed twice to get consistent sampling results
        self.distribution.set_seed(seed)  # for compiling theano functions
        self._set_theano_func()
        self.distribution.set_seed(seed)  # for actual sampling

    def get_params(self):
        params = lasagne.layers.get_all_params(
            self.mean_network, trainable=True)
        return params

    def fprop(self, x, *args, **kwargs):
        """
        Paramaters
        ----------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        Returns
        -------
        mean : Theano variable
            The output of this distribution.
        """

        try:
            inputs = dict(zip(self.given, x))
        except:
            raise ValueError("The length of 'x' must be same as 'given'")

        deterministic = kwargs.pop('deterministic', False)
        mean = lasagne.layers.get_output(
            self.mean_network, inputs, deterministic=deterministic)
        return mean

    def get_input_shape(self):
        """
        Returns
        -------
        tuple
          This represents the shape of the inputs of this distribution.
        """

        return [x.shape for x in self.given]

    def get_output_shape(self):
        """
        Returns
        -------
        tuple
          This represents the shape of the output of this distribution.
        """

        return lasagne.layers.get_output_shape(self.mean_network)

    def sample_given_x(self, x, repeat=1, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        repeat : int or thenao variable

        Returns
        --------
        list
           This contains 'x' and sample ~ p(*|x), such as [x, sample].
        """
        if repeat != 1:
            x = [T.extra_ops.repeat(_x, repeat, axis=0) for _x in x]

        # Feedforward the inputs. The output corresponds to the mean of a distribution,
        # or the mean and the variance if executed from DistributionDouble.
        output = self.fprop(x, **kwargs)
        return [x, self.distribution.sample(*tolist(output))]

    def sample_mean_given_x(self, x, *args, **kwargs):
        """
        Paramaters
        --------
        x : list
           This contains Theano variables, which must to correspond
           to 'given'.

        Returns
        --------
        list
           This contains 'x' and a mean value of sample ~ p(*|x).
        """

        output = self.fprop(x, **kwargs)
        return [x, tolist(output)[0]]

    def log_likelihood_given_x(self, samples, **kwargs):
        """
        Paramaters
        --------
        samples : list
           This contains 'x', which has Theano variables, and test sample.

        Returns
        --------
        Theano variable, shape (n_samples,)
           A log-likelihood, p(sample|x).
        """

        x, sample = samples
        output = self.fprop(x, **kwargs)
        return self.distribution.log_likelihood(sample, *tolist(output))

    def _set_theano_func(self):
        x = self.inputs
        samples = self.fprop(x, deterministic=True)
        self.np_fprop = theano.function(inputs=x,
                                        outputs=samples,
                                        on_unused_input='ignore')

        samples = self.sample_mean_given_x(x, deterministic=True)
        self.np_sample_mean_given_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        samples = self.sample_given_x(x, deterministic=True)
        self.np_sample_given_x = theano.function(
            inputs=x, outputs=samples[-1], on_unused_input='ignore')

        if self.set_log_likelihood:
            sample = self.output
            samples = self.log_likelihood_given_x([x, sample],
                                                  deterministic=True)
            self.np_log_liklihood_given_x = theano.function(
                inputs=x + [sample], outputs=samples,
                on_unused_input='ignore')


class DistributionDouble(Distribution):

    def __init__(self, distribution, mean_network, var_network, given, seed=1):
        self.var_network = var_network
        super(DistributionDouble, self).__init__(
            distribution, mean_network, given, seed)
        if self.get_output_shape() != lasagne.layers.get_output_shape(
                self.var_network):
            raise ValueError("The output shapes of the two networks"
                             "do not match.")

    def get_params(self):
        params = super(DistributionDouble, self).get_params()
        params += self.var_network.get_params(trainable=True)
        # delete duplicated paramaters
        params = sorted(set(params), key=params.index)
        return params

    def fprop(self, x, deterministic=False):
        mean = super(DistributionDouble,
                     self).fprop(x, deterministic=deterministic)
        inputs = dict(zip(self.given, x))
        var = lasagne.layers.get_output(
            self.var_network, inputs, deterministic=deterministic)
        return mean, var


class Deterministic(Distribution):

    def __init__(self, network, given, seed=1):
        distribution = DeterministicSample()
        super(Deterministic, self).__init__(distribution, network, given, seed=seed,
                                            set_log_likelihood=False)


class Bernoulli(Distribution):

    def __init__(self, mean_network, given, temp=0.1, seed=1):
        distribution = BernoulliSample(temp=temp, seed=seed)
        super(Bernoulli, self).__init__(distribution, mean_network, given, seed)


class Categorical(Distribution):

    def __init__(self, mean_network, given, temp=0.1, n_dim=1, seed=1):
        distribution = CategoricalSample(temp=temp, seed=seed)
        self.mean_network = mean_network
        self.n_dim = n_dim
        self.k = self.get_output_shape()[-1]
        super(Categorical, self).__init__(distribution, mean_network, given, seed=seed)

    def sample_given_x(self, x, repeat=1, **kwargs):
        if repeat != 1:
            x = [T.extra_ops.repeat(_x, repeat, axis=0) for _x in x]

        # use fprop of super class
        mean = Distribution.fprop(self, x, **kwargs)
        output = self.distribution.sample(mean).reshape((-1, self.n_dim * self.k))
        return [x, output]

    def fprop(self, x, *args, **kwargs):
        mean = Distribution.fprop(self, x, *args, **kwargs)
        mean = mean.reshape((-1, self.n_dim * self.k))
        return mean


class Gaussian(DistributionDouble):

    def __init__(self, mean_network, var_network, given, seed=1):
        distribution = GaussianSample(seed=seed)
        super(Gaussian, self).__init__(
            distribution, mean_network, var_network, given, seed)


class GaussianConstantVar(Distribution):

    def __init__(self, mean_network, given, var=1, seed=1):
        distribution = GaussianConstantVarSample(constant_var=var, seed=seed)
        super(GaussianConstantVar, self).__init__(distribution, mean_network, given, seed=seed)


class Laplace(DistributionDouble):

    def __init__(self, mean_network, var_network, given, seed=1):
        distribution = LaplaceSample(seed=seed)
        super(Laplace, self).__init__(distribution, mean_network, var_network, given, seed=seed)


class Kumaraswamy(DistributionDouble):
    """
    [Naelisnick+ 2016] Deep Generative Models with Stick-Breaking Priors
    """

    def __init__(self, a_network, b_network,
                 given, stick_breaking=True, seed=1):
        distribution = KumaraswamySample(seed=seed)
        self.stick_breaking = stick_breaking
        super(Kumaraswamy, self).__init__(distribution, a_network, b_network, given, seed=seed)

    def sample_given_x(self, x, repeat=1, **kwargs):
        [x, v] = super(Kumaraswamy, self).sample_given_x(x,
                                                         repeat=repeat,
                                                         **kwargs)
        if self.stick_breaking:
            v = self._stick_breaking_process(v)
        return [x, v]

    def log_likelihood_given_x(self, *args):
        raise NotImplementedError

    def _stick_breaking_process(self, v):
        (n_batch, n_dim) = v.shape

        def segment(v, stick_segment, remaining_stick):
            stick_segment = v * remaining_stick
            remaining_stick *= (1 - v)
            return stick_segment, remaining_stick

        stick_segment = T.zeros((n_batch,))
        remaining_stick = T.ones((n_batch,))

        (stick_segments, remaining_sticks), updates\
            = theano.scan(fn=segment,
                          sequences=v.T[:-1],
                          outputs_info=[stick_segment, remaining_stick])

        return T.concatenate([stick_segments.T,
                              remaining_sticks[-1, :][:, np.newaxis]],
                             axis=1)


class Gamma(DistributionDouble):

    def __init__(self, alpha_network, beta_network, given, seed=1):
        distribution = GammaSample(seed=seed)
        super(Gamma, self).__init__(distribution, alpha_network, beta_network, given, seed=seed)


class Beta(DistributionDouble):

    def __init__(self, alpha_network, beta_network, given,
                 iter_sampling=6, rejection_sampling=True, seed=1):
        distribution = BetaSample(
            iter_sampling=iter_sampling,
            rejection_sampling=rejection_sampling,
            seed=seed)
        super(Beta, self).__init__(distribution, alpha_network, beta_network, given, seed=seed)


class Dirichlet(Distribution):

    def __init__(self, alpha_network, given, k,
                 iter_sampling=6, rejection_sampling=True, seed=1):
        distribution = DirichletSample(k, iter_sampling=iter_sampling,
                                       rejection_sampling=rejection_sampling,
                                       seed=seed)
        self.k = k
        super(Dirichlet, self).__init__(distribution, alpha_network, given, seed=seed)

    def sample_given_x(self, x, repeat=1, **kwargs):
        if repeat != 1:
            x = [T.extra_ops.repeat(_x, repeat, axis=0) for _x in x]

        # use fprop of super class
        mean = Distribution.fprop(self, x, **kwargs)
        _shape = mean.shape
        mean = mean.reshape((_shape[0], _shape[1] / self.k,
                             self.k))
        output = self.distribution.sample(mean).reshape(_shape)
        return [x, output]
