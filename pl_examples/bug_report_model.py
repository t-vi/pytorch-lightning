import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DataParallelPlugin


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        print("batch", batch.shape)
        self.log("train_loss", loss)
        return {"loss": loss, "batch": batch}

    def training_step_end(self, step_outputs):
        losses = step_outputs["loss"]
        print(losses)
        assert len(losses) == 2
        loss = losses.mean()

        batches = step_outputs["batch"]
        assert batches.shape[0] == 2
        assert batches.shape[1] == 32
        print(batches)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():
    train_data = DataLoader(RandomDataset(32, 64), batch_size=4)

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_val_batches=2,
        num_sanity_val_steps=0,
        max_epochs=1,
        weights_summary=None,
        accelerator="dp",
        gpus=2,
    )
    assert isinstance(trainer.training_type_plugin, DataParallelPlugin)
    trainer.fit(model, train_dataloader=train_data)


if __name__ == '__main__':
    run()
