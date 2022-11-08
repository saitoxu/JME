import os
import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import datetime
import requests
from typing import Optional, Callable

from .metrics import calc_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10):
        self.best_value = .0
        self.count = 0
        self.patience = patience


    def __call__(self, value: float):
        should_save, should_stop = False, False
        if value > self.best_value:
            self.best_value = value
            self.count = 0
            should_save = True
        else:
            self.count += 1
        if self.count >= self.patience:
            should_stop = True
        return should_save, should_stop


# ref. https://qiita.com/kaggle_master-arai-san/items/d59b2fb7142ec7e270a5#seed_everything
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def send_notification(start_ts, end_ts, epoch, args):
    arg_list = []
    for key, value in vars(args).items():
        arg_list.append({'key': key, 'value': value})
    headers = {'Content-Type': 'application/json'}
    json = {
        'name': 'KGMB',
        'start': datetime.datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S'),
        'end': datetime.datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S'),
        'duration': end_ts - start_ts,
        'epoch': epoch,
        'args': arg_list
    }
    url = os.environ.get('GAS_URL')
    _ = requests.post(url, headers=headers, json=json)


def evaluate(dataloader, model, Ks, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for u, i, js, dummy_counts in dataloader:
            for idx in range(len(u)):
                dummy_cnt = dummy_counts[idx]
                user = u[idx].to(device)
                items = torch.cat((i.reshape(-1, 1), js), 1).to(device)
                pred = model.predict(user, items[idx])
                pred = torch.cat((pred[0].reshape(-1), pred[dummy_cnt+1:])).cpu()
                preds.append(pred)
    return calc_metrics(preds, Ks)


def distance(triplets):
    assert triplets.size()[1] == 3
    heads = triplets[:, 0, :]
    relations = triplets[:, 1, :]
    tails = triplets[:, 2, :]
    return (heads + relations - tails).norm(dim=1)
    # return 1.0 - F.cosine_similarity(heads + relations, tails)


class MyTripletMarginWithDistanceLoss(nn.Module):
    reduction: str

    def __init__(self, *, distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 swap: bool = False, reduction: str = 'mean'):
        super(MyTripletMarginWithDistanceLoss, self).__init__()
        self.reduction = reduction
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = \
            distance_function if distance_function is not None else nn.PairwiseDistance()
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, margin: Tensor) -> Tensor:
        return triplet_margin_with_distance_loss(anchor, positive, negative,
                                                 distance_function=self.distance_function,
                                                 margin=margin, swap=self.swap, reduction=self.reduction)


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: Tensor,
    *,
    distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    swap: bool = False,
    reduction: str = "mean"
) -> Tensor:
    # if torch.jit.is_scripting():
    #     raise NotImplementedError(
    #         "F.triplet_margin_with_distance_loss does not support JIT scripting: "
    #         "functions requiring Callables cannot be scripted."
    #     )

    # if has_torch_function_variadic(anchor, positive, negative):
    #     return handle_torch_function(
    #         triplet_margin_with_distance_loss,
    #         (anchor, positive, negative),
    #         anchor,
    #         positive,
    #         negative,
    #         distance_function=distance_function,
    #         margin=margin,
    #         swap=swap,
    #         reduction=reduction,
    #     )

    positive_dist = distance_function(anchor, positive)
    negative_dist = distance_function(anchor, negative)

    if swap:
        swap_dist = distance_function(positive, negative)
        negative_dist = torch.min(negative_dist, swap_dist)

    output = torch.clamp(positive_dist - negative_dist + margin, min=0.0)

    if reduction == 'mean':
        return output.mean()
    elif reduction == 'sum':
        return output.sum()
    else:
        return output
