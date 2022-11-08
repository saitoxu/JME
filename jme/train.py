import os
import torch
from torch.utils.data import DataLoader
from time import time

from .parser import parse_args
from .jme import JME
from .dataset import RecDataset, KGDataset, ValOrTestDataset, Phase
from .utils import EarlyStopping, seed_everything, evaluate
from .logger import getLogger


def train(train_rec_dataloader, train_kg_dataloader, model, optimizer, args, device, logger):
    rec_size = len(train_rec_dataloader.dataset)
    model.train()

    kg_size = len(train_kg_dataloader.dataset)
    size = min(rec_size, kg_size)
    # TODO: データ少ない方切り捨てちゃってる
    for batch, (rec_data, kg_data) in enumerate(zip(train_rec_dataloader, train_kg_dataloader)):
        u, i, j, interactions = rec_data
        u, i, j, interactions = u.to(device), i.to(device), j.to(device), interactions.to(device)

        positive_triples, negative_triples = kg_data
        positive_triples, negative_triples = positive_triples.to(device), negative_triples.to(device)

        model.normalize()

        loss = model((u, i, j, interactions), (positive_triples, negative_triples))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(u)
            logger.debug(f"Loss: {loss:>7f} [{current:>8d}/{size:>8d}]")


def test(dataloader, model, Ks, device, logger):
    mrr, hrs, ndcgs = evaluate(dataloader, model, Ks, device)
    rounded_hrs = list(map(lambda x: float(f'{x:>7f}'), hrs))
    rounded_ndcgs = list(map(lambda x: float(f'{x:>7f}'), ndcgs))
    logger.debug(f'MRR:\t{mrr:>7f}')
    logger.debug(f'HRs:\t{rounded_hrs}')
    logger.debug(f'NDCGs:\t{rounded_ndcgs}')
    return hrs[0]


if __name__ == '__main__':
    start = int(time())
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args = parse_args()
    seed_everything(args.seed)

    logger = getLogger(name=__name__, path=args.save_path)

    for key, value in vars(args).items():
        logger.debug(f'{key}: {value}')

    torch.backends.cudnn.benchmark = True

    data_path = f"{args.data_path}/{args.dataset}"
    behavior_data = eval(args.behavior_data)
    num_workers = 2 if os.cpu_count() > 1 else 0

    train_rec_data = RecDataset(data_path=data_path, behavior_data=behavior_data, neg_size=args.neg_size)
    train_rec_dataloader = DataLoader(train_rec_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    train_kg_data = KGDataset(data_path=data_path, neg_size=args.neg_size)
    train_kg_dataloader = DataLoader(train_kg_data, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)

    neg_sample = args.neg_sample == 1
    val_data = ValOrTestDataset(data_path, phase=Phase.VAL, train_data=behavior_data, neg_sample=neg_sample)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    user_entity_map = torch.tensor(train_kg_data.user_entity_map).to(device)
    item_entity_map = torch.tensor(train_kg_data.item_entity_map).to(device)

    model = JME(
        entity_size=train_kg_data.entity_size,
        relation_size=train_kg_data.relation_size,
        user_size=train_rec_data.user_size,
        item_size=train_rec_data.item_size,
        behavior_size=train_rec_data.behavior_size,
        dim=args.dim,
        user_entity_map=user_entity_map,
        item_entity_map=item_entity_map,
        use_behavior_combination=args.use_behavior_combination,
        use_behavior_aware_margin=args.use_behavior_aware_margin,
        use_epl=args.use_epl,
        norm_weight=args.norm_weight,
        device=device
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop = EarlyStopping(args.patience)

    Ks = eval(args.Ks)
    # test(val_dataloader, model, Ks, device, logger)
    epoch = args.epoch
    test_interval = 5
    for t in range(args.epoch):
        logger.debug(f"Epoch {t+1}")
        logger.debug('-'*32)
        train(train_rec_dataloader, train_kg_dataloader, model, optimizer, args, device, logger)
        torch.save(model, args.save_path + 'latest.pth')
        if (t+1) % test_interval == 0:
            hr = test(val_dataloader, model, Ks, device, logger)
            # early stopping
            should_save, should_stop = early_stop(hr)
            if should_save:
                torch.save(model, args.save_path + 'best.pth')
            if should_stop:
                epoch = t + 1
                logger.debug('Early stopping.')
                break
    end = int(time())
    logger.debug('Done!')