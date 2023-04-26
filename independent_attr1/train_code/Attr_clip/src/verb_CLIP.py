# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""
import pdb
import pprint
import numpy as np
import torch
from torch.optim import Adam
import models.losses as losses
import models.optimizer as optim
import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
import utils.metrics as metrics
import utils.misc as misc
from datasets import loader
from datasets.mixup import MixUp
from models import build_model
from utils.meters import EpochTimer, TrainMeter, ValMeter

logger = logging.get_logger(__name__)


# 它接收两个参数logits_image和logits_text分别代表图像和文本的logits，
def info_nce_accuracy(logits_image, logits_text):
    """return acc_image, acc_text"""

    batchsize = logits_image.size(0)

    ground_truth = torch.arange(batchsize, 
        dtype=torch.long, device=logits_image.device)
    # size = [bz, bz]
    # 使用torch.arrange创建一个大小为[bz, bz]的张量ground_truth， 该张量在对角线上具有1，其余位置为0,用于记录正确的预测类别

    # torch.argmax函数找到每个预测结果中概率最大的类别的索引，使用torch.eq函数将其与ground_truth张量进行比较，得到每个预测结果的准确率
    acc_image = ground_truth.eq(torch.argmax(logits_image, dim=-1)).float().mean()
    acc_text  = ground_truth.eq(torch.argmax(logits_text, dim=-1)).float().mean()
    # 具体来说，torch.eq函数返回一个张量，其中每个元素对于两个输入张量中相同位置的元素是否相等，然后将返回的张量转换为浮点型并求平均值。得到每个预测结果的准确率

    return acc_image, acc_text


def train_epoch(
    train_loader,
    model,
    optimizer,         
    scaler,            # 梯度缩放器
    train_meter,       # 训练指标
    cur_epoch,
    cfg,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the `GradScaler` to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()  # 将模型设置为训练模式，开始计时
    data_size = len(train_loader)

    kl_loss_func = losses.KLContrastiveLoss()

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    
    for cur_iter, (inputs, labels, prompts) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        # pdb.set_trace()
        if cfg.NUM_GPUS:
            inputs = inputs.cuda(non_blocking=True)  # bz, frm, ch, r, r
            # labels = labels.cuda()
            prompts = prompts.cuda()                 # 对于每一个batch，将数据传输到GPU设备上，并更新学习率

        if cfg.MIXUP.ENABLE:
            inputs, labels = mixup_fn(inputs, labels)
        
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # troch.cuda.amp.autocast是Pytorch中的一种自动混合精度工具，他可以半自动地将
        # 模型中的某些层或者运算转换为半精度浮点数进行计算，以加速模型的训练和推理，当使用
        # 这段代码时，该上下文管理器会自动将在其范围内的代码块中的浮点操作转换为半精度浮点数
        # 进行计算，从而减少了模型计算所需的内存和计算量，在代码块的结尾，自动混合精度上下文管理器
        # 会自动将结果转换回原始精度，从而保证模型输出的精度与原始精度一致
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            logits_image = model(inputs, prompts)       # 这一句将输入文本和prompt送入模型进行计算
            # Compute the loss.
            loss = kl_loss_func(logits_image, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()   
        scaler.scale(loss).backward()    # 使用梯度缩放器来限制梯度的大小，使其在一定范围内，从而更好地控制模型的更新
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
            preds = preds.detach()
            preds[idx_top1] += preds[idx_top2]
            preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]

        acc_i, acc_t = info_nce_accuracy(logits_image, logits_image.t())
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, acc_i, acc_t = du.all_reduce([loss, acc_i, acc_t])

        # Copy the stats from GPU to CPU (sync point).
        loss, err_i, err_t = (
            loss.item(),
            1-acc_i.item(),
            1-acc_t.item(),
        )

        # Update and log stats.
        train_meter.update_stats(
            err_i,
            err_t,
            loss,
            lr,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()    # 用于在代码块中关闭梯度计算，以减少内存消耗和加速计算
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """

    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    text_features = []              # 记录文本特征

    # print("###### val_loader #####")
    # print(val_loader)
    # print(val_loader.dataset)

    for tokens in val_loader.dataset.prompt_token_per_class:
        if cfg.NUM_GPUS:
            tokens = tokens.cuda()
        # 将token放到CUDA上进行运算的主要原因是因为模型在训练时已经被放到了
        # CUDA上，如果不将token也放到CUDA上，那么在计算模型的输出时，就会
        # 导致模型和输入数据不在同一个设备，从而导致运行时错误
        feat = model(None, tokens)          # 将tokens送入模型进行计算，得到特征向量feat
        text_features.append(feat.mean(0))  # 将对feat矩阵在第0维上进行求平均，即对每个特征维度上的数值取平均值，得到一个大小为[feat_dim]的平均特征向量
    # pdb.set_trace()                         # len(text_features)=413 每个元素是一个512维度的向量, text_features是一个列表
    text_features = torch.stack(text_features) # size = [#class, dim]    此时text_features是一个tensor [413, 512]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)


    for cur_iter, batch in enumerate(val_loader):      # bs=32
        inputs = batch[:-1]     # 将batch中除了最后一个元素以外的所有元素取出来，作为输入数据inputs len(inputs[0])=32 len(input[0][0])=3 inputs最外面为什么是一层列表
        #inputs = inputs[0]      # TODO:权宜之计，以后要修改
        labels = batch[-1]      # 将batch中最后一个元素取出来，作为标签labels
        # pdb.set_trace()
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            inputs = [x.cuda(non_blocking=True) for x in inputs]
            # 设置non_blocking=true可以让数据在异步传输时不会阻塞CPU的运算，从而提高数据传输效率
            labels = labels.cuda()

        val_meter.data_toc()

        preds = 0       # 对模型输入数据的预测结果
        for x in inputs:        # x:[32,3,224,224]   这里对吗
            # pdb.set_trace()
            logits = model(x, text_features)  # [32,413]
            # pdb.set_trace()
            preds = preds + logits            # 
        preds /= len(inputs)                  # len(inputs)=1

        # print("preds=", preds)
        # print("inputs=", inputs)
        # print("labels=", labels)
        

        # select first 1000 IN1K classes for evaluation for IN21k
        if cfg.DATA.IN22k_VAL_IN1K != "":
            preds = preds[:, :1000]
        if cfg.TRAIN.DATASET.lower().startswith("ego4d"):
            preds[:, ~val_loader.dataset.class_mask] = torch.min(preds)

        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1,5))       # 这一步是测量的关键
        # preds [32,413]
        # labels [32]         我这里的label肯定是错了
        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            inputs[0].size(0)
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    # logger.info("Train with config:")
    # logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    # if du.is_master_proc() and cfg.LOG_MODEL_INFO:
    #     misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )
    # cu.load_train_checkpoint函数用于加载之前训练时保存的模型参数、优化器状态和训练轮次等信息
    # 并将他们分别赋值给model, optimizer和scaler

    # Create the train and val loaders.   TODO：trainloader是今天的重点,一定要搞懂怎么使用的
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    # Create meters.   用于记录和统计训练和验证过程中的指标，如训练损失和准确率等
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))


    if cfg.TRAIN.ONLY_VALID:
        eval_epoch(val_loader, model, val_meter, 0, cfg)
        return


    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)


def test(cfg):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)       # 设置numpy的随机数种子
    torch.manual_seed(cfg.RNG_SEED)    # 设置pytorch的随机数种子

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters.
    test_meter = ValMeter(len(test_loader), cfg)
    eval_epoch(test_loader, model, test_meter, -1, cfg)


