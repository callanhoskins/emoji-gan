import time
import datetime

import torch.nn as nn
from torchvision.utils import save_image

from model import Generator, Discriminator
from utils import *
from tqdm import tqdm


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        try:
            torch.nn.init.xavier_uniform_(m.weight)
        except:
            pass
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class Trainer(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # training device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Model hyper-parameters
        self.image_size = config.image_size
        self.im_chan = 3     # standard RGB image
        # self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        ## TODO: Determine n_classes from folder structure
        self.n_classes = config.n_classes
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.n_epoch = config.n_epoch
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        ## TODO: Implement Tensorboard logging
        # self.use_tensorboard = config.use_tensorboard
        self.image_path = config.image_path
        self.save_dir = config.save_dir
        # self.log_path = os.path.join(config.save_dir, 'log')
        self.model_save_path = os.path.join(config.save_dir, 'model')
        self.sample_path = os.path.join(config.save_dir, 'samples')
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        # self.name = config.name

        self.build_model()

        # if self.use_tensorboard:
        #     self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        # # Get saver
        # saver = CheckpointSaver(self.model_save_path,
        #                         max_checkpoints=5,
        #                         metric_name=args.metric_name,
        #                         maximize_metric=args.maximize_metric,
        #                         log=log)

    def train(self):
        # Data iterator
        # data_iter = iter(self.data_loader)
        cur_step = 0
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)

        # Fixed input for debugging
        # fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for epoch in range(self.n_epoch):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            for real_images, labels in self.data_loader:

                # try:
                #     real_images, labels = next(data_iter)
                # except:
                #     data_iter = iter(self.data_loader)
                #     real_images, labels = next(data_iter)

                for _ in range(self.d_iters):

                    # Compute loss with real images
                    # dr1, dr2, df1, df2, gf1, gf2 are attention scores
                    real_images = tensor2var(real_images)
                    one_hot_labels = get_one_hot_labels(labels.to(self.device), self.n_classes)
                    image_one_hot_labels = one_hot_labels[:, :, None, None]
                    image_one_hot_labels = image_one_hot_labels.repeat(1, 1, self.image_size, self.image_size)

                    # apply Gumbel Softmax
                    z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                    # z = torch.rand(real_images.size(0), self.z_dim, device=self.device)
                    # Combine the noise vectors and the one-hot labels for the generator
                    noise_and_labels = combine_vectors(z, one_hot_labels)
                    fake_images, _, _ = self.G(noise_and_labels)
                    fake_image_and_labels = combine_vectors(fake_images.detach(), image_one_hot_labels)
                    d_out_fake, _, _ = self.D(fake_image_and_labels)
                    real_image_and_labels = combine_vectors(real_images, image_one_hot_labels)
                    d_out_real, _, _ = self.D(real_image_and_labels)

                    self.D.zero_grad()

                    if self.adv_loss == 'wgan-gp':
                        d_loss_fake = d_out_fake.mean()
                        d_loss_real = - torch.mean(d_out_real)
                    elif self.adv_loss == 'hinge':
                        d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                        d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                    elif self.adv_loss == 'bce':
                        d_loss_fake = self.criterion(d_out_fake, torch.zeros_like(d_out_fake))
                        d_loss_real = self.criterion(d_out_real, torch.ones_like(d_out_real))

                    # Backward + Optimize
                    d_loss = d_loss_real + d_loss_fake
                    # self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    if self.adv_loss == 'wgan-gp':
                        self.D.zero_grad()

                        # Compute gradient penalty
                        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device).expand_as(real_image_and_labels)
                        interpolated = Variable(alpha * real_image_and_labels.data + (1 - alpha) * fake_image_and_labels.data, requires_grad=True)
                        out, _, _ = self.D(interpolated)

                        grad = torch.autograd.grad(outputs=out,
                                                   inputs=interpolated,
                                                   grad_outputs=torch.ones(out.size()).to(self.device),
                                                   retain_graph=True,
                                                   create_graph=True,
                                                   only_inputs=True)[0]

                        grad = grad.view(grad.size(0), -1)
                        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                        # Backward + Optimize
                        d_loss = self.lambda_gp * d_loss_gp

                        # self.reset_grad()
                        d_loss.backward()
                        self.d_optimizer.step()

                # ================== Train G and gumbel ================== #

                # # Create random noise
                # z = torch.rand(real_images.size(0), self.z_dim, device=self.device)
                # # Combine the noise vectors and the one-hot labels for the generator
                # noise_and_labels = combine_vectors(z, one_hot_labels)
                # fake_images, _, _ = self.G(noise_and_labels)
                # fake_image_and_labels = combine_vectors(fake_images.detach(), image_one_hot_labels)
                g_out_fake, _, _ = self.D(fake_image_and_labels)

                # Compute loss with fake images
                self.G.zero_grad()
                if self.adv_loss == 'wgan-gp':
                    g_loss_fake = - g_out_fake.mean()
                elif self.adv_loss == 'hinge':
                    g_loss_fake = - g_out_fake.mean()
                elif self.adv_loss == 'bce':
                    g_loss_fake = self.criterion(g_out_fake, torch.ones_like(g_out_fake))

                # self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()

                # Print out log info
                if (cur_step + 1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [{}], Step {}, d_loss: {:.4f}, g_loss: {:.4f},"
                          " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                          format(elapsed, cur_step + 1, d_loss.data, g_loss_fake.data,
                                 self.G.attn1.gamma.mean().data, self.G.attn2.gamma.mean().data))

                # Sample images
                if (cur_step + 1) % self.sample_step == 0:
                    # fake_images, _, _ = self.G(fixed_z)
                    save_image(denorm(real_images.data),
                               os.path.join(self.sample_path, '{}_real.png'.format(cur_step + 1)))
                    save_image(denorm(fake_images.data),
                               os.path.join(self.sample_path, '{}_fake.png'.format(cur_step + 1)))

                if (cur_step + 1) % model_save_step == 0:
                    torch.save(self.G.state_dict(),
                               os.path.join(self.model_save_path, '{}_G.pth'.format(cur_step + 1)))
                    torch.save(self.D.state_dict(),
                               os.path.join(self.model_save_path, '{}_D.pth'.format(cur_step + 1)))

                cur_step += 1

    def build_model(self):
        g_input_dim = self.z_dim + self.n_classes
        self.G = Generator(self.image_size, g_input_dim, self.g_conv_dim).to(self.device)
        d_im_chan = self.im_chan + self.n_classes
        self.D = Discriminator(d_im_chan, self.d_conv_dim).to(self.device)
        # Weight initialization
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        if self.adv_loss == 'bce':
            self.criterion = torch.nn.BCEWithLogitsLoss()

        # print networks
        # print(self.G)
        # print(self.D)

    # def build_tensorboard(self):
    #     from logger import Logger
    #     self.logger = Logger(self.log_path)
    #     from torch.utils.tensorboard import SummaryWriter

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
