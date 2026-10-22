import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from torch import nn

from .interpolation import LowFidelityMap


class ResidualGaussianProcess(nn.Module):
    def __init__(self, low, inputs, targets, feature_grid, response_mode,
                 restarts=2):
        super().__init__()
        self.response_mode = response_mode
        self.maps = nn.ModuleList([
            LowFidelityMap(torch.cat([low[:, :2], feature_grid[:, i:i + 1]], 1))
            for i in range(feature_grid.shape[1])])
        self.register_buffer('center', (low[:, :2].amax(0) + low[:, :2].amin(0)) / 2)
        self.register_buffer('scale', ((low[:, :2].amax(0) - low[:, :2].amin(0)) / 2).clamp_min(1e-5))
        self.register_buffer('feature_center', feature_grid.mean(0))
        self.register_buffer('feature_scale', feature_grid.std(0).clamp_min(1e-8))
        features = self.extract_features(inputs)
        values = targets - inputs[:, -1:] if response_mode == 'residual' else targets
        center, scale = values.mean(), values.std().clamp_min(1e-10)
        normalized = (values - center) / scale
        kernel = (ConstantKernel(1., (1e-3, 1e3))
                  * Matern(np.ones(features.shape[1]), (.02, 100.), nu=1.5)
                  + WhiteKernel(1e-4, (1e-10, .1)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                          n_restarts_optimizer=restarts,
                                          random_state=42)
            gp.fit(features.numpy(), normalized.numpy().ravel())
        self.amplitude = float(gp.kernel_.k1.k1.constant_value)
        self.noise = float(gp.kernel_.k2.noise_level) + 1e-10
        self.register_buffer('length_scales', features.new_tensor(gp.kernel_.k1.k2.length_scale))
        self.register_buffer('anchor_features', features)
        covariance = self.kernel(features, features) + self.noise * torch.eye(len(inputs), dtype=inputs.dtype)
        factor = torch.linalg.cholesky(covariance)
        self.register_buffer('cholesky', factor)
        self.register_buffer('coefficient', torch.cholesky_inverse(factor) @ normalized)
        self.register_buffer('response_center', center)
        self.register_buffer('response_scale', scale)
        self.observation_variance = float(scale.square() * self.noise)
        self.evidence = float(gp.log_marginal_likelihood_value_)

    def extract_features(self, inputs):
        coordinates = (inputs[:, :2] - self.center) / self.scale
        values = torch.cat([model(inputs[:, 0], inputs[:, 1]) for model in self.maps], 1)
        values = (values - self.feature_center) / self.feature_scale
        return torch.cat([coordinates, values], 1)

    def kernel(self, left, right):
        distance = torch.cdist(left / self.length_scales, right / self.length_scales)
        scaled = 3 ** .5 * distance
        correlation = (1 + scaled) * (-scaled).exp()
        return self.amplitude * correlation

    def forward(self, inputs):
        cross = self.kernel(self.extract_features(inputs), self.anchor_features)
        prediction = self.response_center + self.response_scale * (cross @ self.coefficient)
        return prediction + inputs[:, -1:] if self.response_mode == 'residual' else prediction

    def posterior_covariance(self, inputs):
        features = self.extract_features(inputs)
        cross = self.kernel(features, self.anchor_features)
        projection = torch.linalg.solve_triangular(self.cholesky, cross.T, upper=False)
        covariance = (self.kernel(features, features) - projection.T @ projection) * self.response_scale.square()
        return (covariance + covariance.T) / 2


class MultifidelityGaussianProcess(nn.Module):
    def __init__(self, low, inputs, targets, feature_grid, logarithmic=False,
                 response_mode='residual', restarts=2):
        super().__init__()
        self.logarithmic = logarithmic
        self.response_mode = response_mode
        self.base = ResidualGaussianProcess(
            self.transform_inputs(low), self.transform_inputs(inputs),
            targets.log() if logarithmic else targets, feature_grid,
            response_mode, restarts)

    def transform_inputs(self, inputs):
        return (torch.cat([inputs[:, :2], inputs[:, -1:].log()], 1)
                if self.logarithmic else inputs)

    def forward(self, inputs):
        prediction = self.base(self.transform_inputs(inputs))
        return prediction.exp() if self.logarithmic else prediction

    def posterior_covariance(self, inputs):
        covariance = self.base.posterior_covariance(self.transform_inputs(inputs))
        if self.logarithmic:
            prediction = self(inputs)
            covariance = prediction * covariance * prediction.T
        return covariance

    def observation_variance(self, inputs):
        prediction = self(inputs).flatten()
        return (prediction.square() * self.base.observation_variance
                if self.logarithmic else torch.full_like(prediction, self.base.observation_variance))
