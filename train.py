#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import csv
import os

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import random_perturb, make_step, inf_data_gen, Logger
from utils import soft_cross_entropy, classwise_loss, LDAMLoss, FocalLoss
from config import *
from data_loader import make_longtailed_imb, get_imbalanced, get_oversampled


LOGNAME = 'Imbalance_' + LOGFILE_BASE
logger = Logger(LOGNAME)
LOGDIR = logger.logdir

LOG_CSV = os.path.join(LOGDIR, f'log_{SEED}.csv')
LOG_CSV_HEADER = [
    'epoch', 'train loss', 'gen loss', 'train acc', 'gen_acc', 'prob_orig', 'prob_targ',
    'test loss', 'major test acc', 'neutral test acc', 'minor test acc', 'test acc', 'f1 score'
]
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(LOG_CSV_HEADER)


def save_checkpoint(acc, model, optim, epoch, index=False):
    # Save checkpoint.
    print('Saving..')

    if isinstance(model, nn.DataParallel):
        model = model.module

    state = {
        'net': model.state_dict(),
        'optimizer': optim.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }

    if index:
        ckpt_name = 'ckpt_epoch' + str(epoch) + '_' + str(SEED) + '.t7'
    else:
        ckpt_name = 'ckpt_' + str(SEED) + '.t7'

    ckpt_path = os.path.join(LOGDIR, ckpt_name)
    torch.save(state, ckpt_path)


def train_epoch(model, criterion, optimizer, data_loader, logger=None):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(data_loader):
        # For SMOTE, get the samples from smote_loader instead of usual loader
        if epoch >= ARGS.warm and ARGS.smote:
            inputs, targets = next(smote_loader_inf)

        """
        inputs: [128, 3, 32, 32]
        targets: [128]
        """
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)

        outputs, _ = model(normalizer(inputs))
        loss = criterion(outputs, targets).mean()

        train_loss += loss.item() * batch_size
        predicted = outputs.max(1)[1]
        total += batch_size
        correct += sum_t(predicted.eq(targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    msg = 'Loss: %.3f| Acc: %.3f%% (%d/%d)' % \
          (train_loss / total, 100. * correct / total, correct, total)
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return train_loss / total, 100. * correct / total


def uniform_loss(outputs):
    weights = torch.ones_like(outputs) / N_CLASSES

    return soft_cross_entropy(outputs, weights, reduction='mean')


def classwise_loss(outputs, targets):
    out_1hot = torch.zeros_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), 1)
    return (outputs * out_1hot).sum(1).mean()


def generation(model_g, model_r, inputs, seed_targets, targets, p_accept,
               gamma, lam, step_size, random_start=True, max_iter=10):
    model_g.eval()
    model_r.eval()
    criterion = nn.CrossEntropyLoss()

    if random_start: # True
        random_noise = random_perturb(inputs, 'l2', 0.5)
        inputs = torch.clamp(inputs + random_noise, 0, 1)

    for _ in range(max_iter):
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs_g, _ = model_g(normalizer(inputs))
        outputs_r, _ = model_r(normalizer(inputs))

        loss = criterion(outputs_g, targets) + lam * classwise_loss(outputs_r, seed_targets) # Equation (2)
        grad, = torch.autograd.grad(loss, [inputs])

        inputs = inputs - make_step(grad, 'l2', step_size)
        inputs = torch.clamp(inputs, 0, 1)

    inputs = inputs.detach()

    outputs_g, _ = model_g(normalizer(inputs))

    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]

    correct = (probs_g >= gamma) * torch.bernoulli(p_accept).byte().to(device)
    model_r.train()

    return inputs, correct


def train_net(model_train, model_gen, criterion, optimizer_train, inputs_orig, targets_orig, gen_idx, gen_targets):
    """
    model_train: f
    model_gen  : g
    inputs_orig:  原始的不平衡的样本
    targets_orig: 原始的不平衡的样本标签
    gen_idx:      从原始样本中选出的要生成的少类的样本 id 
    """
    batch_size = inputs_orig.size(0)

    inputs = inputs_orig.clone()
    targets = targets_orig.clone()

    ########################
    """
    N_SAMPLES_PER_CLASS_T: N_SAMPLES_PER_CLASS_T: [5000., 2997., 1796., 1077.,  645.,  387.,  232.,  139.,   83.,   50.], 每种类别的样本数目
    """
    bs = N_SAMPLES_PER_CLASS_T[targets_orig].repeat(gen_idx.size(0), 1) # e.g. (101, 128),
    gs = N_SAMPLES_PER_CLASS_T[gen_targets].view(-1, 1) # e.g. (101, 1)

    delta = F.relu(bs - gs) # (101, 128)
    p_accept = 1 - ARGS.beta ** delta
    mask_valid = (p_accept.sum(1) > 0) # 在该 batch 中至少有一个样本被选出作为 seed, 该少样本才被选出

    gen_idx = gen_idx[mask_valid]
    gen_targets = gen_targets[mask_valid]
    p_accept = p_accept[mask_valid] # e.g. (90, 128)

    select_idx = torch.multinomial(p_accept, 1, replacement=True).view(-1) # 选出 seed 样本
    p_accept = p_accept.gather(1, select_idx.view(-1, 1)).view(-1)

    seed_targets = targets_orig[select_idx]
    seed_images = inputs_orig[select_idx]

    gen_inputs, correct_mask = generation(model_gen, model_train, seed_images, seed_targets, gen_targets, p_accept,
                                          ARGS.gamma, ARGS.lam, ARGS.step_size, True, ARGS.attack_iter)

    ########################

    # Only change the correctly generated samples
    num_gen = sum_t(correct_mask)
    num_others = batch_size - num_gen

    gen_c_idx = gen_idx[correct_mask]
    others_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    others_mask[gen_c_idx] = 0
    others_idx = others_mask.nonzero().view(-1)

    if num_gen > 0:
        gen_inputs_c = gen_inputs[correct_mask]
        gen_targets_c = gen_targets[correct_mask]

        inputs[gen_c_idx] = gen_inputs_c
        targets[gen_c_idx] = gen_targets_c

    outputs, _ = model_train(normalizer(inputs))
    loss = criterion(outputs, targets)

    optimizer_train.zero_grad()
    loss.mean().backward()
    optimizer_train.step()

    # For logging the training

    oth_loss_total = sum_t(loss[others_idx])
    gen_loss_total = sum_t(loss[gen_c_idx])

    _, predicted = torch.max(outputs[others_idx].data, 1)
    num_correct_oth = sum_t(predicted.eq(targets[others_idx]))

    num_correct_gen, p_g_orig, p_g_targ = 0, 0, 0
    success = torch.zeros(N_CLASSES, 2)

    if num_gen > 0:
        _, predicted_gen = torch.max(outputs[gen_c_idx].data, 1)
        num_correct_gen = sum_t(predicted_gen.eq(targets[gen_c_idx]))
        probs = torch.softmax(outputs[gen_c_idx], 1).data

        p_g_orig = probs.gather(1, seed_targets[correct_mask].view(-1, 1))
        p_g_orig = sum_t(p_g_orig)

        p_g_targ = probs.gather(1, gen_targets_c.view(-1, 1))
        p_g_targ = sum_t(p_g_targ)

    for i in range(N_CLASSES):
        if num_gen > 0:
            success[i, 0] = sum_t(gen_targets_c == i)
        success[i, 1] = sum_t(gen_targets == i)

    return oth_loss_total, gen_loss_total, num_others, num_correct_oth, num_gen, num_correct_gen, p_g_orig, p_g_targ, success


def train_gen_epoch(net_t, net_g, criterion, optimizer, data_loader):
    net_t.train()
    net_g.eval()

    oth_loss, gen_loss = 0, 0
    correct_oth = 0
    correct_gen = 0
    total_oth, total_gen = 1e-6, 1e-6
    p_g_orig, p_g_targ = 0, 0
    t_success = torch.zeros(N_CLASSES, 2)

    for inputs, targets in tqdm(data_loader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(device), targets.to(device)
        """
        inputs: torch.Size([128, 3, 32, 32])
        targets: torch.Size([128])
        """

        # Set a generation target for current batch with re-sampling
        if ARGS.imb_type != 'none':  # Imbalanced, ARGS.imb_type = 'longtail'
            # Keep the sample with this probability
            """
            N_SAMPLES_PER_CLASS_T: [5000., 2997., 1796., 1077.,  645.,  387.,  232.,  139.,   83.,   50.]
            """
            gen_probs = N_SAMPLES_PER_CLASS_T[targets] / N_SAMPLES_PER_CLASS_T[0]
            gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()    # 1 - \frac{N_{y_i}}{N_1}, see Page 4 subsection "Practical implementation vis re-sampling"
            gen_index = gen_index.view(-1)
            gen_targets = targets[gen_index]
        else:   # Balanced
            gen_index = torch.arange(batch_size).view(-1)
            gen_targets = torch.randint(N_CLASSES, (batch_size,)).to(device).long()

        t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success \
            = train_net(net_t, net_g, criterion, optimizer, inputs, targets, gen_index, gen_targets)

        oth_loss += t_loss
        gen_loss += g_loss
        total_oth += num_others
        correct_oth += num_correct
        total_gen += num_gen
        correct_gen += num_gen_correct
        p_g_orig += p_g_orig_batch
        p_g_targ += p_g_targ_batch
        t_success += success

    res = {
        'train_loss': oth_loss / total_oth,
        'gen_loss': gen_loss / total_gen,
        'train_acc': 100. * correct_oth / total_oth,
        'gen_acc': 100. * correct_gen / total_gen,
        'p_g_orig': p_g_orig / total_gen,
        'p_g_targ': p_g_targ / total_gen,
        't_success': t_success
    }

    msg = 't_Loss: %.3f | g_Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_gen: %.3f%% (%d/%d) ' \
          '| Prob_orig: %.3f | Prob_targ: %.3f' % (
        res['train_loss'], res['gen_loss'],
        res['train_acc'], correct_oth, total_oth,
        res['gen_acc'], correct_gen, total_gen,
        res['p_g_orig'], res['p_g_targ']
    )
    if logger:
        logger.log(msg)
    else:
        print(msg)

    return res


if __name__ == '__main__':
    # import pdb; pdb.set_trace()
    TEST_ACC = 0  # best test accuracy
    BEST_VAL = 0  # best validation accuracy

    # Weights for virtual samples are generated
    logger.log('==> Building model: %s' % MODEL)
    """
    MODEL: 'resnet32'
    N_CLASSES: 10
    """
    net = models.__dict__[MODEL](N_CLASSES) # resnet32
    net_seed = models.__dict__[MODEL](N_CLASSES) # resnet32

    net, net_seed = net.to(device), net_seed.to(device)
    optimizer = optim.SGD(net.parameters(), lr=ARGS.lr, momentum=0.9, weight_decay=ARGS.decay)

    if ARGS.resume: # True
        # Load checkpoint.
        logger.log('==> Resuming from checkpoint..')
        ckpt_g = f'./checkpoint/{DATASET}/ratio{ARGS.ratio}/erm_trial1_{MODEL}.t7' # checkpoint/erm_r100_c10_trial1.t7

        """
        net_both: None, 两个网络共同的 pretrained model
        net_t: None, path of network to train
        net_g: './checkpoint/erm_r100_c10_trial1.t7', path of network to generate
        """
        if ARGS.net_both is not None:
            ckpt_t = torch.load(ARGS.net_both)
            net.load_state_dict(ckpt_t['net'])
            optimizer.load_state_dict(ckpt_t['optimizer'])
            START_EPOCH = ckpt_t['epoch'] + 1
            net_seed.load_state_dict(ckpt_t['net2'])
        else: # True
            if ARGS.net_t is not None:
                ckpt_t = torch.load(ARGS.net_t)
                net.load_state_dict(ckpt_t['net'])
                optimizer.load_state_dict(ckpt_t['optimizer'])
                START_EPOCH = ckpt_t['epoch'] + 1

            if ARGS.net_g is not None: # 导入训练好的 g
                ckpt_g = ARGS.net_g
                print(ckpt_g)
                ckpt_g = torch.load(ckpt_g)
                net_seed.load_state_dict(ckpt_g['net'])

    if N_GPUS > 1:
        logger.log('Multi-GPU mode: using %d GPUs for training.' % N_GPUS)
        net = nn.DataParallel(net)
        net_seed = nn.DataParallel(net_seed)
    elif N_GPUS == 1:
        logger.log('Single-GPU mode.')

    """
    ARGS.warm: 160, 先 warm up, 不进行 M2m
    ARGS.over: True
    """
    if ARGS.warm < START_EPOCH and ARGS.over:
        raise ValueError("warm < START_EPOCH")

    """
    EPOCH: 200
    N_CLASSES: 10
    START_EPOCH: 0
    """
    SUCCESS = torch.zeros(EPOCH, N_CLASSES, 2)
    test_stats = {}
    for epoch in range(START_EPOCH, EPOCH):
        logger.log(' * Epoch %d: %s' % (epoch, LOGDIR))

        adjust_learning_rate(optimizer, LR, epoch)

        """
        ARGS.smote: False
        ARGS.over: True
        """
        if epoch == ARGS.warm and ARGS.over:
            if ARGS.smote: # False
                logger.log("=============== Applying smote sampling ===============")
                smote_loader, _, _ = get_smote(DATASET, N_SAMPLES_PER_CLASS, BATCH_SIZE, transform_train, transform_test)
                smote_loader_inf = inf_data_gen(smote_loader)
            else: # True
                logger.log("=============== Applying over sampling ===============")
                """
                get_oversampled: 返回 train_loader, val_loader, test_loader, 其中 train_loader 是 class_balanced 过后的
                """
                train_loader, _, _ = get_oversampled(DATASET, N_SAMPLES_PER_CLASS, BATCH_SIZE,
                                                     transform_train, transform_test) # train_loader, val_loader, test_loader

        ## For Cost-Sensitive Learning ##
        """
        ARGS.cost: False
        """
        if ARGS.cost and epoch >= ARGS.warm: # False
            beta = ARGS.eff_beta
            if beta < 1:
                effective_num = 1.0 - np.power(beta, N_SAMPLES_PER_CLASS)
                per_cls_weights = (1.0 - beta) / np.array(effective_num)
            else:
                per_cls_weights = 1 / np.array(N_SAMPLES_PER_CLASS)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(N_SAMPLES_PER_CLASS)
            per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        else: # True
            per_cls_weights = torch.ones(N_CLASSES).to(device) # default, tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]

        ## Choos a loss function ##

        if ARGS.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights, reduction='none').to(device)
        elif ARGS.loss_type == 'Focal':
            criterion = FocalLoss(weight=per_cls_weights, gamma=ARGS.focal_gamma, reduction='none').to(device)
        elif ARGS.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=N_SAMPLES_PER_CLASS, max_m=0.5, s=30, weight=per_cls_weights,
                                 reduction='none').to(device)
        else:
            raise ValueError("Wrong Loss Type")

        ## Training ( ARGS.warm is used for deferred re-balancing ) ##
        # deferred 推迟的
        if epoch >= ARGS.warm and ARGS.gen:
            train_stats = train_gen_epoch(net, net_seed, criterion, optimizer, train_loader)
            SUCCESS[epoch, :, :] = train_stats['t_success'].float()
            logger.log(SUCCESS[epoch, -10:, :])
            np.save(LOGDIR + '/success.npy', SUCCESS.cpu().numpy())
        else: # 前 160 个 epoch
            import pdb; pdb.set_trace()
            train_loss, train_acc = train_epoch(net, criterion, optimizer, train_loader, logger)
            train_stats = {'train_loss': train_loss, 'train_acc': train_acc}
            # if epoch == 159: 
            if epoch == ARGS.warm - 1: # by Wang Liping 2020-12-23 21:00:52
                save_checkpoint(train_acc, net, optimizer, epoch, True)

        ## Evaluation ##

        val_eval = evaluate(net, val_loader, logger=logger)
        val_acc = val_eval['acc']
        if val_acc >= BEST_VAL:
            BEST_VAL = val_acc

            test_stats = evaluate(net, test_loader, logger=logger)
            TEST_ACC = test_stats['acc']
            TEST_ACC_CLASS = test_stats['class_acc']

            save_checkpoint(TEST_ACC, net, optimizer, epoch)
            logger.log("========== Class-wise test performance ( avg : {} ) ==========".format(TEST_ACC_CLASS.mean()))
            np.save(LOGDIR + '/classwise_acc.npy', TEST_ACC_CLASS.cpu())

        def _convert_scala(x):
            if hasattr(x, 'item'):
                x = x.item()
            return x

        log_tr = ['train_loss', 'gen_loss', 'train_acc', 'gen_acc', 'p_g_orig', 'p_g_targ']
        log_te = ['loss', 'major_acc', 'neutral_acc', 'minor_acc', 'acc', 'f1_score']

        log_vector = [epoch] + [train_stats.get(k, 0) for k in log_tr] + [test_stats.get(k, 0) for k in log_te]
        log_vector = list(map(_convert_scala, log_vector))

        with open(LOG_CSV, 'a') as f:
            logwriter = csv.writer(f, delimiter=',')
            logwriter.writerow(log_vector)

    logger.log(' * %s' % LOGDIR)
    logger.log("Best Accuracy : {}".format(TEST_ACC))
