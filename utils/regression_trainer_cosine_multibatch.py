from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch import optim
# The system protobuf extension can be incompatible with the active libstdc++.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import swin_c_multibatch as models
from datasets.crowd import Crowd
from losses.bay_loss import Bay_Loss
from losses.post_prob import Post_Prob
from math import ceil
import random


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        data_roots = {
            'baseline': {
                'train': os.path.join(args.data_dir, 'train'),
                'val': os.path.join(args.data_dir, 'val'),
            },
            'lora': {
                'train': args.lora_train_dir,
                'val': args.lora_val_dir,
            },
        }
        active_roots = data_roots[args.stage]
        self.datasets = {x: Crowd(active_roots[x],
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        for split in ['train', 'val']:
            if not self.datasets[split].im_list:
                raise ValueError('No .jpg images in {}'.format(active_roots[split]))
        loader_generators = {
            'train': torch.Generator().manual_seed(args.seed),
            'val': torch.Generator().manual_seed(args.seed + 1),
        }
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False),
                                          worker_init_fn=seed_worker,
                                          generator=loader_generators[x])
                            for x in ['train', 'val']}
        model_builder = getattr(models, args.model_name, None)
        if model_builder is None:
            raise ValueError('unknown model: {}'.format(args.model_name))
        # A resumed checkpoint already contains the backbone weights.
        use_pretrained = (args.stage == 'baseline' and args.pretrained_backbone
                          and not args.resume)
        self.model = model_builder(pretrained=use_pretrained,
                                   pretrained_path=args.pretrained_path)
        if args.stage == 'lora':
            if not args.resume:
                checkpoint = load_checkpoint(args.baseline_checkpoint)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    if checkpoint.get('stage', 'baseline') != 'baseline':
                        raise ValueError('LoRA needs a baseline checkpoint')
                    if checkpoint.get('model_name', args.model_name) != args.model_name:
                        raise ValueError('Baseline checkpoint model name does not match')
                state = checkpoint.get('model_state_dict', checkpoint)
                self.model.load_state_dict(state)
            count = self.model.enable_lora(args.lora_rank, args.lora_alpha)
            logging.info('LoRA added to %d Swin attention blocks', count)
        self.model.to(self.device)
        trainable = [parameter for parameter in self.model.parameters()
                     if parameter.requires_grad]
        logging.info('trainable parameters: %d / %d',
                     sum(parameter.numel() for parameter in trainable),
                     sum(parameter.numel() for parameter in self.model.parameters()))
        self.optimizer = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        resumed_state = {}
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = load_checkpoint(args.resume)
                resumed_state = checkpoint
                if checkpoint.get('stage', 'baseline') != args.stage:
                    raise ValueError('Resume checkpoint stage does not match')
                if checkpoint.get('model_name', args.model_name) != args.model_name:
                    raise ValueError('Resume checkpoint model name does not match')
                if args.stage == 'lora' and (
                    checkpoint.get('lora_rank') != args.lora_rank
                    or checkpoint.get('lora_alpha') != args.lora_alpha
                ):
                    raise ValueError('Resume checkpoint LoRA settings do not match')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # New checkpoints store both the human-facing 1-based epoch
                # and its internal 0-based index. Older checkpoints only have
                # the 0-based ``epoch`` field.
                if 'epoch_index' in checkpoint:
                    self.start_epoch = checkpoint['epoch_index'] + 1
                else:
                    self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(load_checkpoint(args.resume))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        # self.criterion = torch.nn.MSELoss(reduction='sum')
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = resumed_state.get('best_mae', np.inf)
        self.best_mse = resumed_state.get('best_mse', np.inf)
        self.best_model_path = resumed_state.get('best_model_path')
        self.save_all = args.save_all
        self.best_count = resumed_state.get('best_count', 0)
        if args.resume and not self.best_model_path:
            previous_best = os.path.join(os.path.dirname(args.resume), 'best_model.pth')
            if os.path.isfile(previous_best):
                self.best_model_path = previous_best
        if self.best_model_path and not os.path.isfile(self.best_model_path):
            raise FileNotFoundError('Saved best model is missing: {}'.format(
                self.best_model_path))
        self.global_step = self.start_epoch * len(self.dataloaders['train'])
        if args.tensorboard_log_interval <= 0:
            raise ValueError('tensorboard_log_interval must be greater than zero')
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))

    def train(self):
        """training process"""
        args = self.args
        try:
            for epoch_index in range(self.start_epoch, args.max_epoch):
                self.epoch_index = epoch_index
                self.epoch = epoch_index + 1
                logging.info(
                    '-'*5 + 'Epoch {}/{}'.format(self.epoch, args.max_epoch) + '-'*5
                )
                self.train_eopch()
                if (
                    self.epoch_index >= args.val_start
                    and (self.epoch_index - args.val_start) % args.val_epoch == 0
                ):
                    self.val_epoch()
                self.save_epoch()
        finally:
            self.writer.close()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_bayesian_loss = AverageMeter()
        epoch_consistency_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        # The LoRA stage keeps the frozen baseline's dropout and stochastic
        # depth in inference mode while gradients update LoRA parameters.
        self.model.train(self.args.stage == 'baseline')

        train_bar = tqdm(
            enumerate(self.dataloaders['train']),
            total=len(self.dataloaders['train']),
            desc='{} {}/{}'.format(
                self.args.stage.title(), self.epoch, self.args.max_epoch),
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for step, (inputs, points, targets, st_sizes) in train_bar:
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs, features = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                bayesian_loss = self.criterion(prob_list, targets, outputs)
                loss_c = outputs.new_zeros(())
                for layer_features in features:
                    # [batch, tokens, channels]. Preserve the original sum over
                    # tokens, but average over the batch so its scale does not
                    # depend on batch size.
                    mean_feature = layer_features.mean(dim=1, keepdim=True)
                    cosine = 1 - F.cosine_similarity(
                        layer_features, mean_feature, dim=2, eps=1e-5
                    )
                    loss_c += cosine.sum(dim=1).mean()
                loss = bayesian_loss + loss_c

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_bayesian_loss.update(bayesian_loss.item(), N)
                epoch_consistency_loss.update(loss_c.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

                if self.global_step % self.args.tensorboard_log_interval == 0:
                    self.writer.add_scalar(
                        'train_step/total_loss', loss.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        'train_step/bayesian_loss',
                        bayesian_loss.item(),
                        self.global_step,
                    )
                    self.writer.add_scalar(
                        'train_step/consistency_loss',
                        loss_c.item(),
                        self.global_step,
                    )

                if (step == 0 or
                        (step + 1) % self.args.tensorboard_log_interval == 0):
                    logging.info(
                        '%s Epoch %d/%d Step %d/%d Train: '
                        'total_loss=%.4f, bayesian_loss=%.4f, '
                        'consistency_loss=%.4f',
                        self.args.stage, self.epoch, self.args.max_epoch,
                        step + 1, len(self.dataloaders['train']),
                        loss.item(), bayesian_loss.item(), loss_c.item(),
                        extra={'file_only': True},
                    )

                train_bar.set_postfix_str(
                    'loss={:.3f}'.format(loss.item()),
                    refresh=False,
                )
                self.global_step += 1

        train_rmse = np.sqrt(epoch_mse.get_avg())
        self.writer.add_scalar('train/total_loss', epoch_loss.get_avg(), self.epoch)
        self.writer.add_scalar(
            'train/bayesian_loss', epoch_bayesian_loss.get_avg(), self.epoch
        )
        self.writer.add_scalar(
            'train/consistency_loss',
            epoch_consistency_loss.get_avg(),
            self.epoch,
        )
        self.writer.add_scalar('train/mae', epoch_mae.get_avg(), self.epoch)
        self.writer.add_scalar('train/rmse', train_rmse, self.epoch)
        self.writer.flush()

        logging.info(
            '{} Epoch {} Train, Loss: {:.4f}, Bayesian Loss: {:.4f}, '
            'Consistency Loss: {:.4f}, RMSE: {:.2f}, MAE: {:.2f}, Cost {:.1f} sec'
            .format(
                self.args.stage,
                self.epoch,
                epoch_loss.get_avg(),
                epoch_bayesian_loss.get_avg(),
                epoch_consistency_loss.get_avg(),
                train_rmse,
                epoch_mae.get_avg(),
                time.time()-epoch_start,
            )
        )

    def save_epoch(self):
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'epoch_index': self.epoch_index,
            'model_name': self.args.model_name,
            'stage': self.args.stage,
            'crop_size': self.args.crop_size,
            'lora_rank': self.args.lora_rank if self.args.stage == 'lora' else 0,
            'lora_alpha': self.args.lora_alpha if self.args.stage == 'lora' else 0,
            'seed': self.args.seed,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse,
            'best_model_path': self.best_model_path,
            'best_count': self.best_count,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        running_abs_error = 0.0
        running_squared_error = 0.0
        val_bar = tqdm(
            self.dataloaders['val'],
            total=len(self.dataloaders['val']),
            desc='{} Val {}/{}'.format(
                self.args.stage.title(), self.epoch, self.args.max_epoch),
            dynamic_ncols=True,
            mininterval=0.5,
        )
        for inputs, count, name in val_bar:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(ceil(1.0 * h / 3584))
                w_stride = int(ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    predicted_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        predicted_count += torch.sum(output).item()
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    predicted_count = torch.sum(outputs).item()

            ground_truth = count[0].item()
            res = ground_truth - predicted_count
            epoch_res.append(res)
            running_abs_error += abs(res)
            running_squared_error += res * res
            processed_samples = len(epoch_res)
            val_bar.set_postfix(
                gt='{:.1f}'.format(ground_truth),
                pred='{:.1f}'.format(predicted_count),
                mae='{:.2f}'.format(running_abs_error / processed_samples),
                rmse='{:.2f}'.format(
                    np.sqrt(running_squared_error / processed_samples)
                ),
                refresh=False,
            )

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        self.writer.add_scalar('val/mae', mae, self.epoch)
        self.writer.add_scalar('val/rmse', mse, self.epoch)
        logging.info('{} Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.args.stage, self.epoch, mse, mae,
                             time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                self.best_model_path = os.path.join(
                    self.save_dir, 'best_model_{}.pth'.format(self.best_count))
                torch.save(model_state_dic, self.best_model_path)
                self.best_count += 1
            else:
                self.best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                torch.save(model_state_dic, self.best_model_path)

        self.writer.add_scalar('best/mae', self.best_mae, self.epoch)
        self.writer.add_scalar('best/rmse', self.best_mse, self.epoch)
        self.writer.flush()
