import math
from argparse import ArgumentParser

import torch
import numpy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from model_checkpoint import ModelCheckpoint
from torch import Tensor, nn
from torch.nn import functional as F

from cifar100_datamodule import CIFAR100DataModule
from tiny_imagenet_datamodule import TinyImagenetDataModule
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pl_bolts.utils.stability import under_review


@under_review()
class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


@under_review()
class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, norm_p=2., mu=1.):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm_p = norm_p
        self.mu = mu

        print(input_dim, output_dim, hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        # print(x.size(), self.hidden_dim, self.input_dim)
        x = self.model(x)
        return F.normalize(x, dim=1, p=self.norm_p) * numpy.sqrt(self.mu)


@under_review()
class SimCLR(LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        dataset: str,
        num_nodes: int = 1,
        arch: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        norm_p: float = 2.0,
        distance_p: float = 2.0,
        gamma: float = 2.0,
        acos_order: int = 0,
        gamma_lambd: float=1.0,
        loss_type: str = "origin",
        projection_mu: float=1.0,
        **kwargs
    ):
        """
        Args:
            batch_size: the batch size
            num_samples: num samples in the dataset
            warmup_epochs: epochs to warmup the lr for
            lr: the optimizer learning rate
            opt_weight_decay: the optimizer weight decay
            loss_temperature: the loss temperature
        """
        super().__init__()
        self.save_hyperparameters()

        self.gpus = gpus
        self.num_nodes = num_nodes
        self.arch = arch
        self.dataset = dataset
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.loss_type = loss_type
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim if self.loss_type != "product" else feat_dim * 2
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.norm_p = norm_p
        self.distance_p = distance_p
        self.gamma = gamma
        self.projection_mu = projection_mu
        self.acos_order = acos_order
        self.max_epochs = max_epochs


        self.optim = optimizer
        self.exclude_bn_bias = exclude_bn_bias
        self.weight_decay = weight_decay
        self.temperature = temperature

        self.start_lr = start_lr
        self.final_lr = final_lr
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.gamma_lambd = gamma_lambd

        print(self.distance_p, self.norm_p, self.feat_dim)
        self.encoder = self.init_model()

        self.projection = Projection(input_dim=512 if self.arch == "resnet18" else 2048, hidden_dim=self.hidden_mlp, output_dim=self.feat_dim, norm_p=self.norm_p, mu=self.projection_mu)

        # compute iters per epoch
        global_batch_size = self.num_nodes * self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

    def init_model(self):
        if self.arch == "resnet18":
            backbone = resnet18
        elif self.arch == "resnet50":
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x):
        # bolts resnet returns a list
        return self.encoder(x)[-1]

    def shared_step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        # final image in tuple is for online eval
        (img1, img2, _), y = batch

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=("bias", "bn")):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optim == "lars":
            optimizer = LARS(
                params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optim == "adam":
            optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def acos_kernel_distance(self, angle):
        if self.acos_order == 1:
            dis = numpy.pi - angle
        elif self.acos_order == 2:
            dis = torch.sin(angle) + (numpy.pi - angle) * torch.cos(angle)
        elif self.acos_order == 3:
            dis = torch.sin(angle) * torch.cos(angle) * 3. + (numpy.pi - angle) * (
                        1 + torch.cos(angle) * torch.cos(angle) * 2.)
        else:
            raise NotImplementedError
        return dis


    def gamma_loss(self, out_1, out_2, gamma, temperature, eps=1e-6):

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2
        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
        cov = torch.pow(torch.cdist(out, out_dist, p=self.distance_p), gamma) * -1.
        # if self.norm_p == 2.0 and self.distance_p == 2.0:
        #     cov = 1 - (cov * 0.5)
        #     # cov2 = torch.mm(out, out_dist.t().contiguous())
        #     # cov3 = cov - cov2
        #     # print(cov3)
        sim = torch.exp(cov / temperature)
        neg = torch.clamp(sim.sum(dim=-1) - sim.diag(), min=eps)
        sim_adj = torch.pow(torch.norm(out_1 - out_2, dim=-1, p=self.distance_p), gamma) * -1.
        # if self.norm_p == 2.0 and self.distance_p == 2.0:
        #     sim_adj = 1 - (sim_adj * 0.5)
        pos = torch.exp(sim_adj / temperature)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / (neg + eps)).mean()
        return loss

    def spectral_loss(self, out_1, out_2, eps=1e-6):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2
            # out: [2 * batch_size, dim]
            # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
        cov = torch.pow(torch.mm(out, out_dist.t().contiguous()), 2)
        pos = torch.sum(torch.clamp(cov.sum(dim=-1) - cov.diag(), min=eps) * (1. / (out_1.shape[0] * (out_1.shape[0] - 1))))
        neg = torch.sum(out_1 * out_2) * (2. / (out_1.shape[0]))
        return pos - neg

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        # if self.distance_p == 2.0:
        #     cov = torch.mm(out, out_dist.t().contiguous())
        #     sim = torch.exp(cov / temperature)
        #     neg = sim.sum(dim=-1)
        #
        # # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        #     row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        #     neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability
        #
        # # Positive similarity, pos becomes [2 * batch_size]
        #     pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        #
        # else:
        if self.acos_order == 0:
            if self.loss_type == "sum":
                loss = self.gamma_loss(out_1=out_1, out_2=out_2, gamma=self.gamma, temperature=self.temperature) * self.gamma_lambd + self.gamma_loss(out_1=out_1, out_2=out_2, gamma=2.0, temperature=self.temperature) * (1. - self.gamma_lambd)
            elif self.loss_type == "origin":
                loss = self.gamma_loss(out_1=out_1, out_2=out_2, gamma=self.gamma, temperature=self.temperature)
            elif self.loss_type == "product":
                loss = self.gamma_loss(out_1=out_1[:, 0:self.feat_dim // 2], out_2=out_2[:, 0:self.feat_dim // 2], gamma=self.gamma,
                                       temperature=self.temperature) * self.gamma_lambd + self.gamma_loss(out_1=out_1[:, self.feat_dim // 2: self.feat_dim],
                                                                                                          out_2=out_2[:, self.feat_dim // 2: self.feat_dim],
                                                                                                          gamma=2.0,
                                                                                                          temperature=self.temperature) * (1. - self.gamma_lambd)
            elif self.loss_type == "spectral":
                loss = self.spectral_loss(out_1=out_1, out_2=out_2)
            else:
                raise NotImplementedError
        else:
            sim = self.acos_kernel_distance(torch.acos(self.temperature * torch.mm(out, out_dist.t().contiguous()) + 1 - self.temperature + eps))
            neg = torch.clamp(sim.sum(dim=-1) - sim.diag(), min=eps)
            pos = self.acos_kernel_distance(torch.acos(self.temperature * torch.sum(out_1 * out_2, dim=-1) + 1 - self.temperature + eps))
            pos = torch.cat([pos, pos], dim=0)
            loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action="store_false")
        parser.add_argument("--maxpool1", action="store_false")
        parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--norm_p", default=2., type=float, help="norm p, -1 for inf")
        parser.add_argument("--distance_p", default=2., type=float, help="distance p, -1 for inf")
        parser.add_argument("--acos_order", default=0, type=int, help="order of acos, 0 for not use acos kernel")
        parser.add_argument("--gamma", default=2., type=float, help="gamma")
        parser.add_argument("--gamma_lambd", default=1., type=float, help="gamma lambd")
        parser.add_argument("--online_ft", action="store_true")
        parser.add_argument("--fp32", action="store_true")

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
        parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
        parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


@under_review()
def cli_main(config, args, isTune=False):
    from ssl_online import SSLOnlineEvaluator
    from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
    from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform

    # parser = ArgumentParser()

    # model args
    # parser = SimCLR.add_model_specific_args(parser)
    # args = parser.parse_args()
    args.__dict__.update(config)

    if args.norm_p == -1.:
        args.norm_p = numpy.inf
    if args.distance_p == -1.:
        args.distance_p = numpy.inf

    if args.dataset == "stl10":
        dm = STL10DataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples

        args.maxpool1 = False
        args.first_conv = True
        args.input_height = dm.dims[-1]

        normalization = stl10_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        val_split = 5000
        if args.num_nodes * args.gpus * args.batch_size > val_split:
            val_split = args.num_nodes * args.gpus * args.batch_size

        dm = CIFAR10DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        ) if args.dataset == "cifar10" else CIFAR100DataModule(
            data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, val_split=val_split
        )

        args.num_samples = dm.num_samples

        args.maxpool1 = False
        args.first_conv = False
        args.input_height = dm.dims[-1]
        # args.temperature = 0.5

        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "imagenet" or args.dataset == "tiny_imagenet":
        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()

        args.gaussian_blur = True
        args.jitter_strength = 1.0

        # args.batch_size = 64
        # args.num_nodes = 8
        # args.gpus = 8  # per-node
        args.max_epochs = 800

        # args.optimizer = "lars"
        # args.learning_rate = 4.8
        # args.final_lr = 0.0048
        # args.start_lr = 0.3
        args.online_ft = True

        dm = ImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers) if args.dataset == "imagenet" else TinyImagenetDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        args.num_samples = dm.num_samples
        args.input_height = dm.dims[-1]
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=args.input_height,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    # print(args)
    model = SimCLR(**args.__dict__)

    online_evaluator = None
    if args.online_ft:
        # online eval
        online_evaluator = SSLOnlineEvaluator(
            drop_p=0.0,
            hidden_dim=None,
            z_dim=args.hidden_mlp,
            num_classes=dm.num_classes,
            dataset=args.dataset,
            isTune=isTune
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor="val_loss")
    callbacks = [] if isTune else [model_checkpoint]
    if args.online_ft:
        callbacks.append(online_evaluator)
    callbacks.append(lr_monitor)

    # print(args.max_steps)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="gpu",
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
    parser.add_argument("--max_t", default=200, type=int, help="max epoch to report")
    parser.add_argument("--num_samples", default=100, type=int, help="number of samples")
    # specify flags to store false
    parser.add_argument("--first_conv", action="store_false")
    parser.add_argument("--maxpool1", action="store_false")
    parser.add_argument("--hidden_mlp", default=2048, type=int, help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
    parser.add_argument("--norm_p", default=2., type=float, help="norm p, -1 for inf")
    parser.add_argument("--distance_p", default=2., type=float, help="distance p, -1 for inf")
    parser.add_argument("--acos_order", default=0, type=int, help="order of acos, 0 for not use acos kernel")
    parser.add_argument("--gamma", default=2., type=float, help="gamma")
    parser.add_argument("--gamma_lambd", default=1., type=float, help="gamma lambd")
    parser.add_argument("--projection_mu", default=1., type=float, help="projection mu")
    parser.add_argument("--loss_type", default="origin", type=str, help="search type, origin, sum , product or spectral")
    parser.add_argument("--online_ft", action="store_true")
    parser.add_argument("--fp32", action="store_true")

    # transform params
    parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
    parser.add_argument("--jitter_strength", type=float, default=1.0, help="jitter strength")
    parser.add_argument("--dataset", type=str, default="cifar10", help="stl10, cifar10")
    parser.add_argument("--data_dir", type=str, default=".", help="path to download data")

    # training params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/lars")
    parser.add_argument("--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay")
    parser.add_argument("--max_epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size per gpu")

    parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
    parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

    args = parser.parse_args()
    cli_main({}, args, isTune=False)
