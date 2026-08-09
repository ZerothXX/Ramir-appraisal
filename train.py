# -*- coding: utf-8 -*-
"""训练主流程（对应 docs/requirements.md 第七章）。

所有超参数一律来自 config.py，本文件不硬编码任何数值/路径。
入口为标准 if __name__ == "__main__"，可直接在 PyCharm 里右键运行。

流程：set_seed -> 加载基座 + processor -> apply_lora -> 数据扫描/划分/DataLoader
-> AdamW（只优化 requires_grad 参数）+ get_scheduler -> 训练循环。

训练循环要点：
* 梯度累积（GRADIENT_ACCUMULATION_STEPS）+ 梯度裁剪（MAX_GRAD_NORM）；
* 混合精度只用 torch.autocast 一套：bf16 -> autocast；fp16 -> autocast + GradScaler；
  no -> 不用。不引入 accelerate；
* step 级（每 LOG_EVERY_N_STEPS 个 optimizer step）与 epoch 级指标都记录；
* 早停（EARLY_STOP_PATIENCE）：验证 loss 连续 N 个 epoch 未创新低则提前停止（0 关闭）；
* checkpoint：epoch{e}_step{s}/ 定期存档 + latest/（最新覆盖）+ best/（val loss 最优）；
* 曲线原始数据与图片分开落盘到 outputs/curve/。
"""

import gc
import math
import os
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

import config
from data import AniTypeDataset, make_collate_fn, scan_dataset, train_val_split
from model import apply_lora, load_base_model_and_processor, save_lora_checkpoint
from utils import ensure_dir, get_logger, plot_curve, save_metrics_json, set_seed

logger = get_logger("train")


def move_to_device(batch):
    """把 batch 中的 tensor 搬到 config.DEVICE（非 tensor 字段原样透传）。"""
    return {k: (v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def run_optimizer_step(optimizer, scaler, trainable_params, max_grad_norm, scheduler):
    """梯度裁剪 + 参数更新 + 调度器步进 + 清梯度，返回裁剪后的梯度范数。

    fp16 时先 scaler.unscale_ 再裁剪（GradScaler 的正确顺序）。
    """
    if scaler is not None:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def evaluate(model, val_loader, autocast_ctx):
    """在验证集上跑一次前向，返回平均 val loss；验证集为空返回 None。"""
    if len(val_loader) == 0:
        logger.warning("验证集为空，跳过本次 eval")
        return None
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch)
            with autocast_ctx:
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1
            del batch
    model.train()
    return total_loss / max(n_batches, 1)


def maybe_save_best(model, val_loss, best_val_loss, save_dir):
    """val loss 更低时覆盖保存 best/ 并返回新最优值；val_loss 为 None 时跳过。"""
    if val_loss is None:
        return best_val_loss
    if val_loss < best_val_loss:
        save_lora_checkpoint(model, save_dir)
        logger.info("新的最优 val loss: %.4f（< %.4f）-> %s", val_loss, best_val_loss, save_dir)
        return val_loss
    return best_val_loss


def main():
    set_seed(config.RANDOM_SEED)
    ensure_dir(config.OUTPUT_TRAIN_MODELS_DIR)
    ensure_dir(config.OUTPUT_CURVE_DIR)

    logger.info("训练超参: epochs=%d batch=%d accum=%d lr=%g wd=%g warmup=%g sched=%s mp=%s "
                "lora_r=%d lora_alpha=%d lora_dropout=%g grad_ckpt=%s",
                config.NUM_EPOCHS, config.BATCH_SIZE, config.GRADIENT_ACCUMULATION_STEPS,
                config.LEARNING_RATE, config.WEIGHT_DECAY, config.WARMUP_RATIO,
                config.LR_SCHEDULER_TYPE, config.MIXED_PRECISION, config.LORA_R,
                config.LORA_ALPHA, config.LORA_DROPOUT, config.GRADIENT_CHECKPOINTING)

    # ---- 模型 ----
    model, processor = load_base_model_and_processor(config)
    model = apply_lora(model, config)
    model.train()

    # ---- 数据 ----
    samples = scan_dataset(config.DATASET_DIR, config.DATASET_DESC_DIR, config.CLOSED_SET_DIRS)
    train_samples, val_samples = train_val_split(
        samples, config.TRAIN_VAL_SPLIT_RATIO, config.RANDOM_SEED)
    logger.info("样本总数: %d，训练: %d，验证: %d", len(samples), len(train_samples), len(val_samples))

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        AniTypeDataset(train_samples, processor),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(
        AniTypeDataset(val_samples, processor),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---- 优化器 / 调度器 ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有任何 requires_grad=True 的参数，请检查 LoRA 配置")
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    micro_steps_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_steps_per_epoch / config.GRADIENT_ACCUMULATION_STEPS)
    # acc_counter 跨 epoch 连续累积，实际 optimizer step 数 = floor(总 micro-batch / accum)；
    # 调度器步数必须与实际步数一致，否则 cosine 调度会在中途截断（原公式算 70 步
    # 而实际只 step 62 次，终段 LR 无法降到预期值）。
    total_steps = max(
        micro_steps_per_epoch * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS, 1)
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_scheduler(
        name=config.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info("每个 epoch %d 个 micro-batch -> %d 个 optimizer step，总步数 %d，warmup %d 步",
                micro_steps_per_epoch, steps_per_epoch, total_steps, warmup_steps)

    # ---- 混合精度（单一机制：torch.autocast；fp16 额外配 GradScaler） ----
    if config.MIXED_PRECISION == "bf16":
        amp_dtype, scaler = torch.bfloat16, None
    elif config.MIXED_PRECISION == "fp16":
        amp_dtype, scaler = torch.float16, torch.cuda.amp.GradScaler()
    elif config.MIXED_PRECISION == "no":
        amp_dtype, scaler = None, None
    else:
        raise ValueError(f"config.MIXED_PRECISION 取值非法: {config.MIXED_PRECISION}"
                         f"（应为 'no' / 'fp16' / 'bf16'）")
    if amp_dtype is not None:
        autocast_ctx = torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype)
    else:
        autocast_ctx = nullcontext()

    # ---- 指标记录 ----
    metrics = {
        "step": [],            # optimizer step 序号
        "loss": [],            # step 级平均 train loss
        "lr": [],              # step 级当前学习率
        "grad_norm": [],       # step 级梯度范数（裁剪后）
        "val_loss": [],        # step 级验证 loss（未验证的 step 为 None）
        "epoch": [],           # epoch 序号
        "epoch_train_loss": [],  # epoch 级平均 train loss
        "epoch_val_loss": [],    # epoch 级验证 loss
    }
    best_val_loss = float("inf")
    global_step = 0
    acc_counter = 0  # 全局累积窗口计数（跨 epoch 连续累积，不跨 epoch 重置）
    epochs_without_improvement = 0  # 早停：val loss 未创新低的连续 epoch 数
    early_stop_triggered = False

    # ---- 训练循环 ----
    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_loss_sum = 0.0   # epoch 级真实 loss 之和
        epoch_micro_steps = 0
        window_loss_sum = 0.0  # 当前 optimizer step 窗口内 display loss 之和
        window_micro_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}", unit="batch")
        for batch in pbar:
            batch = move_to_device(batch)
            with autocast_ctx:
                outputs = model(**batch)
            # 除以累积步数，使多步累积的梯度等于平均梯度
            loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            display_loss = loss.item()  # 显示用（= 真实 loss / accum）
            epoch_loss_sum += display_loss * config.GRADIENT_ACCUMULATION_STEPS
            epoch_micro_steps += 1
            window_loss_sum += display_loss
            window_micro_steps += 1
            acc_counter += 1

            if acc_counter % config.GRADIENT_ACCUMULATION_STEPS == 0:
                global_step += 1
                grad_norm = run_optimizer_step(optimizer, scaler, trainable_params,
                                               config.MAX_GRAD_NORM, scheduler)
                window_avg = window_loss_sum / window_micro_steps

                metrics["step"].append(global_step)
                metrics["loss"].append(window_avg)
                metrics["lr"].append(scheduler.get_last_lr()[0])
                metrics["grad_norm"].append(grad_norm)
                metrics["val_loss"].append(None)

                if global_step % config.LOG_EVERY_N_STEPS == 0:
                    logger.info("Step %d/%d loss=%.4f lr=%.2e grad_norm=%.4f",
                                global_step, total_steps, window_avg,
                                scheduler.get_last_lr()[0], grad_norm)

                # checkpoint：定期存档（epoch 末还会再存一次）
                if global_step % config.SAVE_STEPS == 0:
                    save_lora_checkpoint(
                        model, os.path.join(
                            config.OUTPUT_TRAIN_MODELS_DIR,
                            f"epoch{epoch}_step{global_step}"))
                    save_lora_checkpoint(
                        model, os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, "latest"))

                # step 级验证（可选，做了就记录到 metrics）
                if global_step % config.EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader, autocast_ctx)
                    metrics["val_loss"][-1] = val_loss
                    logger.info("Step %d eval: val_loss=%s", global_step,
                                f"{val_loss:.4f}" if val_loss is not None else "None")
                    best_val_loss = maybe_save_best(
                        model, val_loss, best_val_loss,
                        os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, "best"))
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                window_loss_sum = 0.0
                window_micro_steps = 0

            pbar.set_postfix(loss=f"{display_loss * config.GRADIENT_ACCUMULATION_STEPS:.4f}")
            del batch

        # ---- epoch 级指标：平均 train loss + 一次验证 ----
        epoch_train_loss = epoch_loss_sum / max(epoch_micro_steps, 1)
        val_loss = evaluate(model, val_loader, autocast_ctx)
        metrics["epoch"].append(epoch)
        metrics["epoch_train_loss"].append(epoch_train_loss)
        metrics["epoch_val_loss"].append(val_loss)
        logger.info("Epoch %d 完成: train_loss=%.4f val_loss=%s", epoch, epoch_train_loss,
                    f"{val_loss:.4f}" if val_loss is not None else "None")

        # ---- 早停判断（EARLY_STOP_PATIENCE>0 时启用）----
        # 在 maybe_save_best 更新 best 之前比较，val loss 未创新低则连续计数，
        # 连续达到 PATIENCE 个 epoch 即提前停止（best/ 仍保留历史最优）。
        early_stopped_this_epoch = False
        if config.EARLY_STOP_PATIENCE > 0 and val_loss is not None:
            if val_loss < best_val_loss:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                    logger.info("早停触发：val loss 连续 %d 个 epoch 未下降"
                                "（当前 %.4f，历史最优 %.4f），提前结束训练",
                                config.EARLY_STOP_PATIENCE, val_loss, best_val_loss)
                    early_stopped_this_epoch = True

        best_val_loss = maybe_save_best(
            model, val_loss, best_val_loss,
            os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, "best"))
        # 每 epoch 存一次带编号的 checkpoint + 覆盖 latest/
        save_lora_checkpoint(
            model, os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, f"epoch{epoch}_step{global_step}"))
        save_lora_checkpoint(model, os.path.join(config.OUTPUT_TRAIN_MODELS_DIR, "latest"))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if early_stopped_this_epoch:
            early_stop_triggered = True
            break

    # ---- 落盘：原始数据 + 曲线图 ----
    metrics["early_stop_patience"] = config.EARLY_STOP_PATIENCE
    metrics["early_stop_triggered"] = early_stop_triggered
    save_metrics_json(metrics, os.path.join(config.OUTPUT_CURVE_DIR, "metrics.json"))
    plot_curve(
        metrics["step"], {"train_loss": metrics["loss"]},
        "Training loss per optimizer step", "Step", "Loss",
        os.path.join(config.OUTPUT_CURVE_DIR, "loss_per_step.png"))
    plot_curve(
        metrics["epoch"], {"train_loss": metrics["epoch_train_loss"],
                           "val_loss": metrics["epoch_val_loss"]},
        "Train/Val loss per epoch", "Epoch", "Loss",
        os.path.join(config.OUTPUT_CURVE_DIR, "loss_per_epoch.png"))
    plot_curve(
        metrics["step"], {"lr": metrics["lr"]},
        "Learning rate per optimizer step", "Step", "LR",
        os.path.join(config.OUTPUT_CURVE_DIR, "learning_rate.png"))
    logger.info("指标原始数据与曲线已保存到 %s", config.OUTPUT_CURVE_DIR)
    logger.info("训练完成。最优 val loss: %s", f"{best_val_loss:.4f}" if best_val_loss < float("inf") else "None")


if __name__ == "__main__":
    main()
