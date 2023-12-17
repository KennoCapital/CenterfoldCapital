import torch
YEAR_FLOAT = 1.0
MONTH_FLOAT = 1.0 / 12.0


def float_to_time_str(x: float or torch.Tensor) -> str:
    if x >= YEAR_FLOAT:
        return '{:.4g}Y'.format(float(x))
    if x in (MONTH_FLOAT * y for y in range(1, 12)):
        return '{:.1g}M'.format(float(x) / MONTH_FLOAT)


def float_to_notional_str(x: float or torch.Tensor) -> str:
    return 'N {:,.10g}'.format(float(x))
