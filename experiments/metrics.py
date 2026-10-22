import numpy as np


def regression_metrics(prediction, truth, scale, drag=False):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    error = prediction - truth
    relative = np.abs(error[truth != 0] / truth[truth != 0])
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.abs(error).mean()),
                nrmse=float(np.sqrt(np.mean(error ** 2)) / scale),
                p95_absolute_error=float(np.quantile(np.abs(error), .95)),
                max_absolute_error=float(np.abs(error).max()),
                mean_absolute_relative_percent=float(relative.mean() * 100) if len(relative) else None,
                relative_above10_percent_count=int((relative > .1).sum()),
                undefined_relative_count=int((truth == 0).sum()),
                negative_prediction_count=int((prediction < 0).sum()) if drag else None)


def normalized_aulc(states):
    budgets = [state['budget'] for state in states]
    return float(np.trapezoid([state['mean_nrmse'] for state in states], budgets)
                 / (budgets[-1] - budgets[0]))
