import torch


class LowFidelityMap(torch.nn.Module):

    def __init__(self, low):
        super().__init__()
        low = low.detach().cpu().double()
        (alpha, mach) = (low[:, 0].unique(sorted=True), low[:, 1].unique(sorted=True))
        values = torch.empty((len(alpha), len(mach)), dtype=torch.float64)
        values[torch.searchsorted(alpha, low[:, 0].contiguous()), torch.searchsorted(mach, low[:, 1].contiguous())] = low[:, 2]
        h = alpha.diff()[:, None]
        secant = values.diff(dim=0) / h
        slopes = torch.empty_like(values)
        (slopes[0], slopes[-1]) = (secant[0], secant[-1])
        if len(alpha) > 2:
            (w1, w2) = (2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1])
            same_sign = secant[:-1] * secant[1:] > 0
            safe_left = torch.where(same_sign, secant[:-1], torch.ones_like(secant[:-1]))
            safe_right = torch.where(same_sign, secant[1:], torch.ones_like(secant[1:]))
            slopes[1:-1] = torch.where(same_sign, (w1 + w2) / (w1 / safe_left + w2 / safe_right), 0)
        for (name, value) in (('alpha', alpha), ('mach', mach), ('values', values), ('slopes', slopes)):
            self.register_buffer(name, value)

    def forward(self, alpha, mach):
        (alpha, mach) = (alpha.double().reshape(-1), mach.double().reshape(-1))
        upper = torch.searchsorted(self.alpha, alpha.contiguous()).clamp(1, len(self.alpha) - 1)
        lower = upper - 1
        h = self.alpha[upper] - self.alpha[lower]
        t = (alpha - self.alpha[lower]) / h

        def curve(column):
            (left, right) = (self.values[lower, column], self.values[upper, column])
            value = (2 * t ** 3 - 3 * t ** 2 + 1) * left + (t ** 3 - 2 * t ** 2 + t) * h * self.slopes[lower, column] + (-2 * t ** 3 + 3 * t ** 2) * right + (t ** 3 - t ** 2) * h * self.slopes[upper, column]
            value = torch.where(alpha < self.alpha[0], self.values[0, column] + (alpha - self.alpha[0]) * self.slopes[0, column], value)
            return torch.where(alpha > self.alpha[-1], self.values[-1, column] + (alpha - self.alpha[-1]) * self.slopes[-1, column], value)
        if len(self.mach) == 1:
            return curve(torch.zeros_like(lower))[:, None]
        high_mach = torch.searchsorted(self.mach, mach.contiguous()).clamp(1, len(self.mach) - 1)
        low_mach = high_mach - 1
        fraction = (mach - self.mach[low_mach]) / (self.mach[high_mach] - self.mach[low_mach])
        return (curve(low_mach) * (1 - fraction) + curve(high_mach) * fraction)[:, None]
