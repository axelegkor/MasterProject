import torch
from path import dataset_paths


def construct_coarse2fine_loader(cfg):

    if cfg.DATASET.NAME == 'FakeNews':
        from .fake_news_bert import construct_fake_news_splits
        train, val, test, fine_classes, coarse_classes = construct_fake_news_splits(cfg)

    else:
        raise NotImplementedError('Dataset not implemented: {}'.format(cfg.DATASET.NAME))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=cfg.SOLVER.DEVICES,
        rank=cfg.RANK_ID,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=cfg.SOLVER.BATCH_SIZE,
                                               shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=cfg.DATALOADER.PIN_MEMORY, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    return train_loader, val_loader, test_loader, fine_classes, coarse_classes