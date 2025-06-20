from datasets import FakeNewsBertData
from datasets.utils.wrappers import NeighborsWrapper

def construct_fake_news_splits(cfg):
    file = cfg.DATASET.DATAROOT

    train_ds = FakeNewsBertData(file, two_views=cfg.DATASET.MULTIVIEW)
    val_ds = FakeNewsBertData(file)
    test_ds = FakeNewsBertData(file)

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)

    print(
        f"Number of training samples: {len(train_ds)}, number of validation samples: {len(val_ds)}, number of test samples: {len(test_ds)}"
    )

    fine_classes = (
        train_ds.num_fine
        if hasattr(train_ds, "num_fine")
        else train_ds.dataset.num_fine
    )
    coarse_classes = (
        train_ds.num_coarse
        if hasattr(train_ds, "num_coarse")
        else train_ds.dataset.num_coarse
    )

    return train_ds, val_ds, test_ds, fine_classes, coarse_classes
