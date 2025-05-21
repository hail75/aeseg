from functools import partial

import torch.nn.functional as F


nonlinearity = partial(F.relu, inplace=True)