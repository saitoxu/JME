import torch
from torch.utils.data import DataLoader

from .parser import parse_args
from .dataset import ValOrTestDataset, Phase
from .metrics import calc_metrics
from .utils import seed_everything, evaluate


def test(dataloader, model, Ks, device):
    mrr, hrs, ndcgs = evaluate(dataloader, model, Ks, device)
    rounded_hrs = list(map(lambda x: float(f'{x:>7f}'), hrs))
    # rounded_ndcgs = list(map(lambda x: float(f'{x:>7f}'), ndcgs))
    # print(f'MRR:\t{mrr:>7f}')
    # print(f'HRs:\t{rounded_hrs}')
    # print(f'NDCGs:\t{rounded_ndcgs}')
    rounded_ndcgs = list(map(lambda x: str(float(f'{x:>7f}')), ndcgs))
    print(f'{mrr:>7f},{",".join(rounded_ndcgs)}')


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    data_path = f'{args.data_path}/{args.dataset}'
    all_behavior_data = ['train_view.txt', 'train_fav.txt', 'train.txt']
    # GHCFとの比較の場合
    # all_behavior_data = ['train.txt']

    phase = Phase.VAL if args.val == 1 else Phase.TEST
    neg_sample = args.neg_sample == 1
    test_data = ValOrTestDataset(data_path, phase=phase, train_data=all_behavior_data, neg_sample=neg_sample)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    model = torch.load(args.model_path)

    Ks = eval(args.Ks)
    test(test_dataloader, model, Ks, device)