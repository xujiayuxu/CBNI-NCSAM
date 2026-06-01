import argparse

import torch
from model.smooth_cross_entropy import smooth_crossentropy
from model.presnet import PreResNet34
from utility.log import Log
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.scheduler import CosineScheduler
from data.cifar_fsam import get_datasets_cutout

from sam import SAM
import time
import random
from utility.time_record import TIME_RECORD
from utility.save_file import write_to_file, copy_files_to_folders, sivefile_config
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from data.noise_ind_datasets import cifar_dataloader

# 新增导入：用于实例依赖噪声模拟 (CBN)
from sklearn.cluster import KMeans


def initialize(args, seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True

    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(filepath, model, optimizer, scheduler):
    """加载模型、优化器和 epoch 状态"""
    if not torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(filepath,weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    print(f"--- 成功加载模型，将从 Epoch {start_epoch} 开始训练 ---")
    return start_epoch

def extract_features(model, inputs, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        features = model(inputs, return_features=True)
        features = F.normalize(features, dim=1)
    if was_training:
        model.train()
    return features.cpu().numpy()





def simulate_idn_labels_cbn(model, inputs, logits, labels, args, epoch, device, flip_ratio=0.2, num_clusters=5):

    features = extract_features(model, inputs, device)  # (batch_size, feat_dim)
    num_clusters = max(1, min(num_clusters, len(features)))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(features)
    clusters = kmeans.labels_  # (batch_size,)


    C = logits.shape[1]  # 类数
    noise_matrices = []
    for k in range(num_clusters):
        cluster_idx = np.where(clusters == k)[0]
        if len(cluster_idx) == 0:
            T_k = np.eye(C) * (1 - flip_ratio) + (flip_ratio / (C - 1)) * (1 - np.eye(C))
            noise_matrices.append(T_k)
            continue

        cluster_logits = logits[cluster_idx]

        top2_values, _ = torch.topk(cluster_logits, 2, dim=1)
        delta = top2_values[:, 0] - top2_values[:, 1]
        max_delta = torch.max(delta).clamp_min(1e-12)
        noise_rate_k = flip_ratio * (1 - torch.mean(delta) / max_delta)
        noise_rate_k = noise_rate_k.clamp(min=0.0, max=flip_ratio)


        T_k = np.eye(C) * (1 - noise_rate_k.item()) + (noise_rate_k.item() / (C - 1)) * (1 - np.eye(C))
        noise_matrices.append(T_k)


    sim_labels = labels.clone().cpu().numpy()
    for i in range(len(labels)):
        k = clusters[i]
        T = noise_matrices[k]
        true_label = sim_labels[i]
        sim_labels[i] = np.random.choice(C, p=T[true_label])

    sim_labels = torch.from_numpy(sim_labels).long().to(device)

    progress = epoch / args.epochs
    scale = (3 - 2 * progress) * (progress ** 2) * 2
    if scale > 1:
        scale = 1

    return sim_labels, scale


def train(nora):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=28, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.05, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=0.05, type=int, help="Rho parameter for SAM.")
    # parser.add_argument("--rho_max", default=0.1, type=int, help="Rho parameter for SAM.")
    # parser.add_argument("--rho_min", default=0.1, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=10, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model", default='Presnet34', type=str, help="resnet18, wideresnet, pyramidnet")
    parser.add_argument("--datasets", default='cifar-100', type=str, help="CIFAR10, cifar100,CIFAR10_noise")
    parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='../results_cac/')
    parser.add_argument('--noise_ratio', type=float, help='corruption rate, should be less than 1', default=0.2)
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
    parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric]', default='ins')
    parser.add_argument('--transforms', type=str, default="false")
    parser.add_argument("--flip_interval", type=int, default=1)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--flip_ratio", type=float, default=0.2, help="标签反转率")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to a checkpoint file to resume training from.")

    # 参数：用于IDN模拟
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of clusters for CBN.")  #表示：在每个 batch 里，把当前样本的特征向量分成多少个簇。
    parser.add_argument("--noise_grad_k", type=float, default=5.0,
                        help="Scale factor for the simulated noisy-label gradient correction.")

    args = parser.parse_args()
    # --- 新增代码：打印所有参数 ---
    print("----------- Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print(f"{arg}: {value}")
    print("---------------------------------------------")
    # ------------------------------------
    # sam resnet18: 200,128,0.05,0.001,0.05
    # sam WideResNet: 200,128,0.05,0.001,0.1

    save_file_list, save_file_dir = sivefile_config("results/", args.datasets, args.model,
                                                    "sam_noise_" + str(args.noise_ratio))

    # index_num = random.randint(1, 2000)
    index_num = 1
    print('Seed:', index_num)
    write_to_file("other/whole_train_time.txt", 'Seed:' + str(index_num))
    initialize(args, seed=index_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    class_num = 100

    log = Log(log_each=10)
    #train_loader, test_loader = get_datasets_cutout(args)
    if args.datasets=="cifar-100" or args.datasets=="cifar-10":
        dataloaders = cifar_dataloader(cifar_type=args.datasets,
                                       root='/home/xjy/code/Label_noise_experiment/data',
                                       batch_size=args.batch_size,
                                       num_workers=0,
                                       noise_type=args.noise_type,
                                       percent=args.noise_ratio)
        train_loader, test_loader = dataloaders.run(
            mode='train_index'), dataloaders.run(mode='test')
    elif args.datasets=="CIFAR10_noise" or args.datasets=="CIFAR100_noise":   #对称噪声和非对称噪声
        train_loader, test_loader = get_datasets_cutout(args)





    test_acc_history = []

    model = PreResNet34(num_class=class_num,low_dim=20).to(device)
    #model = resnet34(num_classes=class_num).to(device)

    base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    scheduler = CosineScheduler(T_max=args.epochs * len(train_loader), max_value=args.learning_rate, min_value=0.0,
                                optimizer=base_optimizer)
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho,
                    adaptive=args.adaptive, lr=args.learning_rate)

    whole_time = 0
    timer = TIME_RECORD()

    criterion = torch.nn.CrossEntropyLoss(reduce=False).cuda()
    criterion11 = torch.nn.CrossEntropyLoss().cuda()
    start_epoch = 1
    if args.resume_checkpoint:
        try:
            start_epoch = load_checkpoint(args.resume_checkpoint, model, optimizer, scheduler)

            total_steps = args.epochs * len(train_loader)
            current_step = (start_epoch - 1) * len(train_loader)

            scheduler = CosineScheduler(T_max=total_steps, max_value=args.learning_rate, min_value=0.0,
                                        optimizer=base_optimizer)

            for _ in range(current_step):
                scheduler.step()

            print(f"--- 学习率调度器已重新初始化并推进到第 {current_step} 步 ---")
        except FileNotFoundError:
            print(f"--- 警告：未找到 Checkpoint 文件 {args.resume_checkpoint}，从 Epoch 1 开始训练 ---")


    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        start_time = time.time()
        timer.time_dict = {}
        in_warmup = epoch < args.warmup_epochs
        do_flip = (not in_warmup) and (epoch % args.flip_interval == 0)

        for index, batch in enumerate(tqdm(train_loader)):
            #inputs, targets = batch[0][1].to(device), batch[1].long().to(device)  # (b.to(device) for b in batch)  cifar10
            inputs, targets = batch[0][1].to(device), batch[1].long().to(device)
            if in_warmup:
                if index == 0:
                    print("--- warmup用SGD更新 ---")
                # --- Warmup阶段：只用普通SGD ---
                base_optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.mean().backward()
                base_optimizer.step()  # 普通SGD更新
                scheduler.step()

                continue

                # 跳过后面的SAM更新部分
                # if index == 0:
                #     print("--- warmup纯SAM更新 ---")
                # # --- Warmup阶段：只用普通SAM ---
                # enable_running_stats(model)
                # predictions = model(inputs)
                # loss = criterion(predictions, targets)
                #
                # loss.mean().backward()
                #
                # optimizer.first_step(zero_grad=True)
                # disable_running_stats(model)
                # predictions = model(inputs)
                # lossf = criterion(predictions, targets)
                #
                # lossf.mean().backward()
                # optimizer.second_step(zero_grad=True)
                #
                # with torch.no_grad():
                #     scheduler.step()
                # continue  # 跳过后面的SAM更新部分

            # 先计算logits (用于模拟，如果需要)
            outputs = model(inputs)
            logits = outputs  # 假设outputs是logits

            noise_grads = []
            if do_flip:
                if index == 0:
                    print("--- 开始实例依赖噪声模拟（CBN方法） ---")

                noisy_labels, scale = simulate_idn_labels_cbn(model, inputs, logits, targets, args, epoch, device,
                                                              flip_ratio=args.flip_ratio, num_clusters=args.num_clusters)
                selected = torch.nonzero(noisy_labels != targets).squeeze(1)  # 选中的索引
                num_flip = selected.numel()
                if index == 0:
                    print(f"--- 模拟了 {num_flip} 个噪声样本 ---")

                if num_flip > 0:
                    # 仅用模拟噪声样本计算噪声梯度 gn
                    optimizer.zero_grad()
                    out_noisy_sel = model(inputs[selected])
                    loss_noisy_sel = smooth_crossentropy(out_noisy_sel, noisy_labels[selected],
                                                         smoothing=args.label_smoothing)
                    loss_noisy_sel.mean().backward()

                    noise_grads = [
                        (p, p.grad.detach().clone())
                        for p in model.parameters()
                        if p.grad is not None
                    ]

                if index == 0:
                    print(f"[Epoch {epoch}] ΔW scale = {scale:.3f}",
                          f"k={args.noise_grad_k}",
                          f"lr*k={args.learning_rate * args.noise_grad_k:.3f}")

            optimizer.zero_grad()
            enable_running_stats(model)
            predictions= model(inputs)
            loss = criterion(predictions, targets)

            loss.mean().backward()

            optimizer.first_step(zero_grad=True)
            # ---- 修正扰动 (W + e → W + e - ΔW = W + e + lr*gn) ----
            with torch.no_grad():
                if do_flip and noise_grads:
                    for p, gn in noise_grads:
                        dW = - args.learning_rate * gn * scale * args.noise_grad_k
                        p.data -= dW  # 注意 ΔW 已经是 -lr*gn，因此这里仍是 p.data -= dW

            disable_running_stats(model)
            predictions = model(inputs)
            lossf = criterion(predictions, targets)

            lossf.mean().backward()
            optimizer.second_step(zero_grad=True)
            total_train_loss += lossf.mean().item()
            train_batches += 1

            with torch.no_grad():
                scheduler.step()

        end_time = time.time()
        es_time = end_time - start_time
        whole_time += es_time
        write_to_file("other/whole_train_time.txt", str(whole_time))

        model.eval()
        log.eval(len_dataset=len(test_loader))

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch[0].to(device), batch[1].to(device)  # (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = criterion(predictions, targets)
                # loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                log(model, loss.cpu(), correct.cpu())
            log.flush()
            test_acc_history.append(log.acc)
            write_to_file("other/accuracy-noise+sam+dw-cifar10-f0.4.txt", str(log.acc))

    copy_files_to_folders(save_file_list, save_file_dir)

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    # global test_err, test_loss

    total_loss = 0
    total_err = 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            total_loss += loss.item() * input_var.shape[0]
            total_err += (output.max(dim=1)[1] != target_var).sum().item()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))


    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    aaa = [0.2]
    for j in range(1):
        for i in range(1):
            train(aaa[i])
