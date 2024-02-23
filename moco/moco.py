import copy
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from lightly.data import LightlyDataset
# from lightly.loss import NTXentLoss
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--loss_type', default='origin', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--gamma', default=1.0, type=float, metavar='M',
                    help='mce gamma')
parser.add_argument('--temperature', default=0.1, type=float)
parser.add_argument('--temperature2', default=0.3, type=float)
parser.add_argument('--gamma_lambd', default=0.2, type=float)

args = parser.parse_args()

num_workers = 8
batch_size = 512
memory_bank_size = 4096
seed = 1
max_epochs = args.epochs


class NTXentLoss(MemoryBankModule):
    def __init__(
        self,
        temperature: float = 0.5,
        memory_bank_size: int = 0,
        gather_distributed: bool = False,
    ):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8
        self.gamma_lambd = args.gamma_lambd
        self.temperature = args.temperature
        self.gamma = args.gamma
        self.temperature2 = args.temperature2
        self.loss_type = args.loss_type
        # print('wocaonima')

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def gamma_loss(self, out_1, out_2, gamma, temperature, eps=1e-6, negative=None):

        cov = torch.pow(torch.cdist(out_1, negative.T, p=2), gamma) * -1.
        sim = torch.exp(cov / temperature)
        neg = torch.clamp(sim.sum(dim=-1), min=eps)
        sim_adj = torch.pow(torch.norm(out_1 - out_2, dim=-1, p=2.), gamma) * -1.
        pos = torch.exp(sim_adj / temperature)
        loss = -torch.log(pos / (neg + eps)).mean()
      
        return loss

    def nt_xent_loss(self, out_1, out_2, negative):
            """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
            """

            if self.loss_type == "sum":
                loss = self.gamma_loss(out_1=out_1, out_2=out_2, gamma=self.gamma,
                                           temperature=self.temperature, negative=negative) * self.gamma_lambd + self.gamma_loss(
                        out_1=out_1,
                        out_2=out_2,
                        gamma=2.0,
                        temperature=self.temperature2, negative=negative) * (
                                   1. - self.gamma_lambd)
            elif self.loss_type == "origin":
                loss = self.gamma_loss(out_1=out_1, out_2=out_2, gamma=self.gamma, temperature=self.temperature, negative=negative)
            else:
                raise NotImplementedError

            return loss

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):

        device = out0.device
        batch_size, _ = out0.shape
      
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        out1, negatives = super(NTXentLoss, self).forward(
            out1, update=out0.requires_grad
        )
        return self.nt_xent_loss(out0, out1, negative=negatives.to(device))

# %%
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from Kaggle
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# The dataset structure should be like this:
# cifar10/train/
#  L airplane/
#    L 10008_airplane.png
#    L ...
#  L automobile/
#  L bird/
#  L cat/
#  L deer/
#  L dog/
#  L frog/
#  L horse/
#  L ship/
#  L truck/
path_to_train = "/data/cifar10/cifar10/train/"
path_to_test = "/data/cifar10/cifar10/test/"
pl.seed_everything(seed)

transform = MoCoV2Transform(
    input_size=32,
    gaussian_blur=0.0,
)
train_classifier_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)


if args.dataset == 'cifar10':
# We use the moco augmentations for training moco
    dataset_train_moco = LightlyDataset(input_dir=path_to_train, transform=transform)

    dataset_train_classifier = LightlyDataset(
        input_dir=path_to_train, transform=train_classifier_transforms
    )

    dataset_test = LightlyDataset(input_dir=path_to_test, transform=test_transforms)
elif args.dataset == 'cifar100':
    dataset_train_moco = LightlyDataset.from_torch_dataset(
        torchvision.datasets.cifar.CIFAR100(root='/data', transform=transform, train=True, download=False), transform=transform)
    dataset_train_classifier = LightlyDataset.from_torch_dataset(
        torchvision.datasets.cifar.CIFAR100(root='/data', transform=train_classifier_transforms, train=True, download=False), transform=train_classifier_transforms)

    dataset_test = LightlyDataset.from_torch_dataset(
        torchvision.datasets.cifar.CIFAR100(root='/data', transform=test_transforms, train=False, download=False), transform=test_transforms)
elif args.dataset == 'tiny':
    dataset_train_moco = LightlyDataset(input_dir='/data/tiny-imagenet-200/train', transform=transform)
    dataset_train_classifier = LightlyDataset(
        input_dir='/data/tiny-imagenet-200/train', transform=train_classifier_transforms
    )

    dataset_test = LightlyDataset(input_dir='/data/tiny-imagenet-200/val', transform=test_transforms)
else:
    raise NotImplementedError

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18", 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = NTXentLoss(temperature=0.1, memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class Classifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        if args.dataset == 'cifar10': self.fc = nn.Linear(512, 10)
        elif args.dataset == 'cifar100': self.fc = nn.Linear(512, 100)
        else: self.fc = nn.Linear(512, 200)

        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def on_train_epoch_end(self):
        self.custom_histogram_weights()

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        self.validation_step_outputs.append((num, correct))
        return num, correct

    def on_validation_epoch_end(self):
        # calculate and log top1 accuracy
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100)
        return [optim], [scheduler]


model = MocoModel()
trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu")
trainer.fit(model, dataloader_train_moco)

model.eval()
classifier = Classifier(model.backbone)
trainer = pl.Trainer(max_epochs=100, devices=1, accelerator="gpu")
trainer.fit(classifier, dataloader_train_classifier, dataloader_test)
