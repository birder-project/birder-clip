import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
from birder.common import cli
from birder.common import training_utils
from birder.common.lib import format_duration
from birder.conf import settings
from birder.data.dataloader.webdataset import make_wds_loader
from birder.data.datasets.directory import get_image_loader
from birder.data.datasets.webdataset import wds_args_from_info
from birder.data.transforms.classification import get_rgb_stats
from birder.data.transforms.classification import inference_preset
from birder.data.transforms.classification import training_preset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from birder_clip.common import fs_ops
from birder_clip.common import lib
from birder_clip.common import training_cli
from birder_clip.common import training_utils as clip_training_utils
from birder_clip.data.datasets.csv import ImageTextCsvDataset
from birder_clip.data.datasets.fake import FakeImageTextData
from birder_clip.data.datasets.webdataset import make_wds_dataset
from birder_clip.data.datasets.webdataset import prepare_wds_args
from birder_clip.loss import CLIPLoss
from birder_clip.model_registry import registry
from birder_clip.net.base import get_image_text_signature
from birder_clip.tokenizers import get_tokenizer

logger = logging.getLogger(__name__)


def train(args: argparse.Namespace) -> None:
    #
    # Initialize
    #
    device, device_id, disable_tqdm = training_utils.init_training(args, logger)

    if args.size is None:
        args.size = registry.get_default_size(args.network, image_encoder=args.image_encoder)

    logger.info(f"Using size={args.size}")

    #
    # Data
    #
    rgb_stats = get_rgb_stats(args.rgb_mode, args.rgb_mean, args.rgb_std)
    logger.debug(f"Using RGB stats: {rgb_stats}")

    training_transform = training_preset(
        args.size,
        args.aug_type,
        args.aug_level,
        rgb_stats,
        args.resize_min_scale,
        args.re_prob,
        args.use_grayscale,
        args.ra_num_ops,
        args.ra_magnitude,
        args.augmix_severity,
        args.simple_crop,
    )
    val_transform = inference_preset(args.size, rgb_stats, args.center_crop, args.simple_crop)

    if args.tokenizer is None:
        args.tokenizer = registry.get_default_tokenizer(args.network)
    if args.tokenizer is None:
        args.tokenizer = "simple_tokenizer"

    if args.context_length is None:
        args.context_length = registry.get_default_context_length(args.network, tokenizer=args.tokenizer)

    logger.info(f"Using context length={args.context_length}")
    tokenizer = get_tokenizer(args.tokenizer, context_length=args.context_length)

    if args.use_fake_data is True:
        logger.warning("Using fake data")
        training_dataset = FakeImageTextData(
            10000, (args.channels, *args.size), num_classes=10, transform=training_transform, tokenizer=tokenizer
        )
        validation_dataset = FakeImageTextData(
            1000, (args.channels, *args.size), num_classes=10, transform=val_transform, tokenizer=tokenizer
        )

    elif args.wds is True:
        training_wds_path: str | list[str]
        val_wds_path: str | list[str]
        if args.wds_info is not None:
            training_wds_path, training_size = wds_args_from_info(args.wds_info, args.wds_training_split)
            val_wds_path, val_size = wds_args_from_info(args.wds_info, args.wds_val_split)
            if args.wds_train_size is not None:
                training_size = args.wds_train_size
            if args.wds_val_size is not None:
                val_size = args.wds_val_size
        else:
            training_wds_path, training_size = prepare_wds_args(args.data_path[0], args.wds_train_size, device)
            if args.val_path is not None:
                val_wds_path, val_size = prepare_wds_args(args.val_path[0], args.wds_val_size, device)

        training_dataset = make_wds_dataset(
            training_wds_path,
            dataset_size=training_size,
            shuffle=True,
            samples_names=True,
            transform=training_transform,
            image_decoder=args.img_loader,
            channels=args.channels,
            tokenizer=tokenizer,
            cache_dir=args.wds_cache_dir,
        )
        if args.wds_info is not None or args.val_path is not None:
            validation_dataset = make_wds_dataset(
                val_wds_path,
                dataset_size=val_size,  # pylint: disable=possibly-used-before-assignment
                shuffle=False,
                samples_names=True,
                transform=val_transform,
                image_decoder=args.img_loader,
                channels=args.channels,
                tokenizer=tokenizer,
                cache_dir=args.wds_cache_dir,
            )
        else:
            validation_dataset = None

    else:
        image_loader = get_image_loader(args.img_loader, args.channels)
        training_dataset = ImageTextCsvDataset(
            args.data_path, transforms=training_transform, tokenizer=tokenizer, loader=image_loader
        )
        if args.val_path is not None:
            validation_dataset = ImageTextCsvDataset(
                args.val_path, transforms=val_transform, tokenizer=tokenizer, loader=image_loader
            )
        else:
            validation_dataset = None

    logger.info(f"Using device {device}:{device_id}")
    logger.info(f"Training dataset has {len(training_dataset):,} samples")
    if validation_dataset is not None:
        logger.info(f"Validation dataset has {len(validation_dataset):,} samples")

    batch_size: int = args.batch_size
    grad_accum_steps: int = args.grad_accum_steps
    model_ema_steps: int = args.model_ema_steps
    logger.debug(f"Effective batch size = {batch_size * grad_accum_steps * args.world_size}")

    # Data loaders and samplers
    virtual_epoch_mode = args.steps_per_epoch is not None
    train_sampler, validation_sampler = training_utils.get_samplers(
        args, training_dataset, validation_dataset, infinite=virtual_epoch_mode
    )

    if args.wds is True:
        training_loader = make_wds_loader(
            training_dataset,
            batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            collate_fn=None,
            world_size=args.world_size,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
            persistent_workers=args.persistent_workers,
            shuffle=args.wds_extra_shuffle,
            infinite=virtual_epoch_mode,
        )
        if validation_dataset is not None:
            validation_loader = make_wds_loader(
                validation_dataset,
                batch_size,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                collate_fn=None,
                world_size=args.world_size,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
            )
        else:
            validation_loader = None

    else:
        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            drop_last=args.drop_last,
            persistent_workers=args.persistent_workers,
        )
        if validation_dataset is not None:
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                sampler=validation_sampler,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                pin_memory=args.pin_memory,
                persistent_workers=args.persistent_workers,
            )
        else:
            validation_loader = None

    if virtual_epoch_mode is True:
        optimizer_steps_per_epoch = args.steps_per_epoch
        epoch_num_batches = args.steps_per_epoch * grad_accum_steps
        epoch_samples = epoch_num_batches * batch_size * args.world_size
        logger.debug(f"Virtual epoch has {epoch_samples:,} samples")
    else:
        optimizer_steps_per_epoch = math.ceil(len(training_loader) / grad_accum_steps)
        epoch_num_batches = len(training_loader)
        epoch_samples = len(training_dataset)

    assert args.model_ema is False or model_ema_steps <= optimizer_steps_per_epoch

    last_batch_idx = epoch_num_batches - 1
    last_accum_steps = epoch_num_batches % grad_accum_steps
    if last_accum_steps == 0:
        last_accum_steps = grad_accum_steps

    last_accum_start_idx = epoch_num_batches - last_accum_steps
    begin_epoch = 1
    epochs = args.epochs + 1
    args.stop_epoch = training_utils.normalize_stop_epoch(epochs, args.stop_epoch)

    logger.debug(
        f"Epoch has {epoch_num_batches} iterations ({optimizer_steps_per_epoch} steps), "
        f"virtual mode={virtual_epoch_mode}"
    )

    #
    # Initialize network
    #
    model_dtype: torch.dtype = getattr(torch, args.model_dtype)
    sample_shape = (batch_size, args.channels, *args.size)  # B, C, H, W
    text_shape = (batch_size, tokenizer.context_length)
    network_name = lib.get_image_text_network_name(
        args.network,
        tag=args.tag,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
    )

    registered_config = registry.all_nets[args.network.lower()].config  # type: ignore[misc]
    model_config = lib.get_image_text_model_config(
        registered_config,
        args.model_config,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        tokenizer=args.tokenizer,
        image_encoder_config=args.image_encoder_config,
        text_encoder_config=args.text_encoder_config,
        input_channels=args.channels,
        image_size=args.size,
        context_length=tokenizer.context_length,
    )

    if args.resume_epoch is not None:
        begin_epoch = args.resume_epoch + 1
        net, checkpoint_rgb_stats, training_states = fs_ops.load_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
            image_encoder=args.image_encoder,
            text_encoder=args.text_encoder,
            embed_dim=args.embed_dim,
            tokenizer=args.tokenizer,
            image_encoder_config=args.image_encoder_config,
            text_encoder_config=args.text_encoder_config,
            epoch=args.resume_epoch,
            new_size=args.size,
            new_context_length=args.context_length,
            strict=not args.non_strict_weights,
        )
        if checkpoint_rgb_stats != rgb_stats:
            logger.warning(
                f"Resuming training with RGB stats {rgb_stats}, "
                f"but checkpoint was saved with {checkpoint_rgb_stats}"
            )

    elif args.pretrained is True:
        fs_ops.download_model_by_weights(network_name, progress_bar=training_utils.is_local_primary(args))
        net, checkpoint_rgb_stats, training_states = fs_ops.load_checkpoint(
            device,
            args.network,
            config=args.model_config,
            tag=args.tag,
            image_encoder=args.image_encoder,
            text_encoder=args.text_encoder,
            embed_dim=args.embed_dim,
            tokenizer=args.tokenizer,
            image_encoder_config=args.image_encoder_config,
            text_encoder_config=args.text_encoder_config,
            epoch=None,
            new_size=args.size,
            new_context_length=args.context_length,
            strict=not args.non_strict_weights,
        )
        if checkpoint_rgb_stats != rgb_stats:
            logger.warning(
                f"Training with RGB stats {rgb_stats}, "
                f"but pretrained checkpoint was saved with {checkpoint_rgb_stats}"
            )

    else:
        net = registry.net_factory(args.network, config=model_config)
        training_states = fs_ops.TrainingStates.empty()

    net.to(device, dtype=model_dtype)
    if args.freeze_bn is True:
        net = training_utils.freeze_batchnorm2d(net)
    elif args.sync_bn is True and args.distributed is True:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.fast_matmul is True or args.amp is True:
        torch.set_float32_matmul_precision("high")

    # Compile network
    if args.compile is True:
        net = torch.compile(net, fullgraph=args.compile_fullgraph, mode=args.compile_mode)

    #
    # Loss criteria, optimizer, learning rate scheduler and training parameter groups
    #

    # Loss criteria
    if args.loss == "clip":
        criterion = CLIPLoss()
    else:
        raise ValueError("Unsupported loss")

    # Learning rate scaling
    lr = training_utils.scale_lr(args)

    # Training parameter groups
    custom_keys_weight_decay = training_utils.get_wd_custom_keys(args)
    custom_layer_weight_decay = args.custom_layer_wd
    if args.transformer_embedding_decay is not None:
        custom_layer_weight_decay = {"token_embedding": args.transformer_embedding_decay}
        if args.custom_layer_wd is not None:
            custom_layer_weight_decay.update(args.custom_layer_wd)

    parameters = training_utils.optimizer_parameter_groups(
        net,
        args.wd,
        base_lr=lr,
        norm_weight_decay=args.norm_wd,
        custom_keys_weight_decay=custom_keys_weight_decay,
        custom_layer_weight_decay=custom_layer_weight_decay,
        bias_lr=args.bias_lr,
    )

    if args.lr_scheduler_update == "epoch":
        step_update = False
        scheduler_steps_per_epoch = 1
    elif args.lr_scheduler_update == "step":
        step_update = True
        scheduler_steps_per_epoch = optimizer_steps_per_epoch
    else:
        raise ValueError("Unsupported lr_scheduler_update")

    # Optimizer and learning rate scheduler
    optimizer = training_utils.get_optimizer(parameters, lr, args)
    scheduler = training_utils.get_scheduler(optimizer, scheduler_steps_per_epoch, args)
    if args.compile_opt is True:
        optimizer.step = torch.compile(optimizer.step, fullgraph=False)

    # Gradient scaler and AMP related tasks
    scaler, amp_dtype = training_utils.get_amp_scaler(args.amp, args.amp_dtype)

    # Load states
    if args.load_states is True:
        optimizer.load_state_dict(training_states.optimizer_state)
        scheduler.load_state_dict(training_states.scheduler_state)
        if scaler is not None:
            scaler.load_state_dict(training_states.scaler_state)

    elif args.load_scheduler is True:
        scheduler.load_state_dict(training_states.scheduler_state)
        last_lrs = scheduler.get_last_lr()
        for g, last_lr in zip(optimizer.param_groups, last_lrs):
            g["lr"] = last_lr

    last_lr = float(max(scheduler.get_last_lr()))
    if args.plot_lr is True:
        logger.info("Fast forwarding scheduler...")
        optimizer.step()
        lrs = []
        for _ in range(begin_epoch, epochs):
            for _ in range(scheduler_steps_per_epoch):
                lrs.append(float(max(scheduler.get_last_lr())))
                scheduler.step()

        plt.plot(
            np.linspace(begin_epoch, epochs, scheduler_steps_per_epoch * (epochs - begin_epoch), endpoint=False),
            lrs,
        )
        plt.show()
        raise SystemExit(0)

    #
    # Distributed (DDP) and Model EMA
    #
    if args.model_ema_warmup is not None:
        ema_warmup_steps = args.model_ema_warmup * optimizer_steps_per_epoch
    elif args.warmup_epochs is not None:
        ema_warmup_steps = args.warmup_epochs * optimizer_steps_per_epoch
    elif args.warmup_steps is not None:
        ema_warmup_steps = args.warmup_steps
    else:
        ema_warmup_steps = 0

    logger.debug(f"EMA warmup steps = {ema_warmup_steps}")
    net_without_ddp = net
    no_sync_cm = nullcontext
    if args.distributed is True:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.local_rank],
            find_unused_parameters=args.find_unused_parameters,
            broadcast_buffers=not args.no_broadcast_buffers,
        )
        no_sync_cm = net.no_sync
        net_without_ddp = net.module

    if args.model_ema is True:
        model_base = net_without_ddp
        model_ema = training_utils.ema_model(args, net_without_ddp, device=device)
        if args.load_states is True and training_states.ema_model_state is not None:
            logger.info("Setting model EMA weights...")
            if args.compile is True and hasattr(model_ema.module, "_orig_mod") is True:
                model_ema.module._orig_mod.load_state_dict(training_states.ema_model_state)
            else:
                model_ema.module.load_state_dict(training_states.ema_model_state)

            model_ema.n_averaged += 1  # pylint:disable=no-member

        model_to_save = model_ema.module
        eval_model = model_ema

    else:
        model_base = None
        model_to_save = net_without_ddp
        eval_model = net

    if args.compile is True and hasattr(model_to_save, "_orig_mod") is True:
        model_to_save = model_to_save._orig_mod
    if args.compile is True and hasattr(model_base, "_orig_mod") is True:
        model_base = model_base._orig_mod  # type: ignore[union-attr]

    #
    # Misc
    #

    # Print network summary
    net_for_info = net_without_ddp
    if args.compile is True and hasattr(net_without_ddp, "_orig_mod") is True:
        net_for_info = net_without_ddp._orig_mod

    if args.no_summary is False:
        summary = torchinfo.summary(
            net_for_info,
            device=device,
            input_data=[
                torch.rand(sample_shape, device=device, dtype=model_dtype),
                torch.zeros(text_shape, device=device, dtype=torch.long),
            ],
            col_names=["input_size", "output_size", "kernel_size", "num_params"],
            depth=4,
            verbose=0,
        )
        if training_utils.is_global_primary(args) is True:
            # Write to stderr, same as all the logs
            print(summary, file=sys.stderr)

    # Training logs
    training_log_path = training_utils.training_log_path(network_name, device, args.experiment)
    logger.info(f"Logging training run at {training_log_path}")
    summary_writer = SummaryWriter(training_log_path)

    signature = get_image_text_signature(sample_shape, tokenizer.context_length)
    file_handler: logging.Handler = logging.NullHandler()
    if training_utils.is_global_primary(args) is True:
        with torch.no_grad():
            summary_writer.add_graph(
                net_for_info,
                (
                    torch.rand(sample_shape, device=device, dtype=model_dtype),
                    torch.zeros(text_shape, device=device, dtype=torch.long),
                ),
            )

        summary_writer.flush()
        fs_ops.write_config(network_name, net_for_info, signature=signature, rgb_stats=rgb_stats)
        file_handler = clip_training_utils.setup_file_logging(training_log_path.joinpath("training.log"))
        training_utils.write_training_args_json(training_log_path, args)
        training_utils.write_training_data_json(
            training_log_path,
            {
                "training_samples": len(training_dataset),
                "validation_samples": len(validation_dataset) if validation_dataset is not None else None,
            },
        )

    #
    # Training loop
    #
    optimizer_step = (begin_epoch - 1) * optimizer_steps_per_epoch
    if virtual_epoch_mode is True:
        train_iter = iter(training_loader)

    running_loss = training_utils.SmoothedValue(window_size=64)
    running_val_loss = training_utils.SmoothedValue()

    logger.info(f"Starting training with learning rate of {last_lr}")
    epoch = begin_epoch - 1
    for epoch in range(begin_epoch, args.stop_epoch):
        tic = time.time()
        net.train()

        # Clear metrics
        running_loss.clear()
        running_val_loss.clear()

        if args.distributed is True or virtual_epoch_mode is True:
            train_sampler.set_epoch(epoch)

        progress = tqdm(
            desc=f"Epoch {epoch}/{epochs-1}",
            total=epoch_samples,
            leave=False,
            disable=disable_tqdm,
            unit="samples",
            initial=0,
        )

        # Zero the parameter gradients
        optimizer.zero_grad()

        epoch_start = time.time()
        start_time = epoch_start
        last_idx = -1
        batch_iter: Iterator[tuple[int, Any]]
        if virtual_epoch_mode is True:
            batch_iter = ((i, next(train_iter)) for i in range(epoch_num_batches))
        else:
            batch_iter = enumerate(training_loader)

        for i, (_, images, texts) in batch_iter:
            images = images.to(device, dtype=model_dtype, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            optimizer_update = (i == last_batch_idx) or ((i + 1) % grad_accum_steps == 0)
            sync_context = no_sync_cm if optimizer_update is False else nullcontext
            if i >= last_accum_start_idx:
                effective_accum_steps = last_accum_steps
            else:
                effective_accum_steps = grad_accum_steps

            # Forward and backward
            with sync_context():
                with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                    model_out = net(images, texts, return_features=True)
                    losses = criterion(
                        image_features=model_out["image_features"],
                        text_features=model_out["text_features"],
                        logit_scale=model_out["logit_scale"],
                        logit_bias=model_out["logit_bias"],
                    )
                    raw_loss = sum(losses.values())

                loss = raw_loss / effective_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if optimizer_update is True:
                if scaler is not None:
                    if args.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    if args.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)

                    optimizer.step()

                optimizer.zero_grad()
                with torch.no_grad():
                    # Clamp as done at OpenCLIP
                    training_utils.unwrap_compiled_module(net_without_ddp).logit_scale.clamp_(0, math.log(100.0))

                if step_update is True:
                    scheduler.step()

            if optimizer_update is True:
                optimizer_step += 1

            # Exponential moving average
            if args.model_ema is True and optimizer_update is True and optimizer_step % model_ema_steps == 0:
                model_ema.update_parameters(net)
                if ema_warmup_steps > 0 and optimizer_step <= ema_warmup_steps:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)  # pylint: disable=no-member

            # Statistics
            running_loss.update(raw_loss.detach())

            # Write statistics
            if (i + 1) % args.log_interval == 0 or i == last_batch_idx:
                time_now = time.time()
                time_cost = time_now - start_time
                iters_processed_in_interval = i - last_idx
                rate = iters_processed_in_interval * (batch_size * args.world_size) / time_cost

                avg_time_per_iter = time_cost / iters_processed_in_interval
                remaining_iters_in_epoch = last_batch_idx - i
                estimated_time_to_finish_epoch = remaining_iters_in_epoch * avg_time_per_iter

                start_time = time_now
                last_idx = i
                cur_lr = float(max(scheduler.get_last_lr()))

                running_loss.synchronize_between_processes(device)

                with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                    log.info(
                        f"[Trn] Epoch {epoch}/{epochs-1}, iter {i+1}/{last_batch_idx+1}  "
                        f"Loss: {running_loss.avg:.4f}  "
                        f"Elapsed: {format_duration(time_now-epoch_start)}  "
                        f"ETA: {format_duration(estimated_time_to_finish_epoch)}  "
                        f"T: {time_cost:.1f}s  "
                        f"R: {rate:.1f} samples/s  "
                        f"LR: {cur_lr:.4e}"
                    )

                if training_utils.is_global_primary(args) is True:
                    summary_writer.add_scalars(
                        "loss",
                        {"training": running_loss.avg},
                        ((epoch - 1) * epoch_samples) + ((i + 1) * batch_size * args.world_size),
                    )

            # Update progress bar
            progress.update(n=batch_size * args.world_size)

        progress.close()

        # Epoch training metrics
        running_loss.synchronize_between_processes(device)
        logger.info(f"[Trn] Epoch {epoch}/{epochs-1} training_loss: {running_loss.global_avg:.4f}")

        # Validation
        epoch_val_loss = None
        if validation_loader is not None and validation_dataset is not None:
            eval_model.eval()
            progress = tqdm(
                desc=f"Epoch {epoch}/{epochs-1}",
                total=len(validation_dataset),
                leave=False,
                disable=disable_tqdm,
                unit="samples",
                initial=0,
            )
            with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                log.info(f"[Val] Starting validation for epoch {epoch}/{epochs-1}...")

            epoch_start = time.time()
            with torch.inference_mode():
                for _, images, texts in validation_loader:
                    images = images.to(device, dtype=model_dtype, non_blocking=True)
                    texts = texts.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                        model_out = eval_model(images, texts, return_features=True)
                        losses = criterion(
                            image_features=model_out["image_features"],
                            text_features=model_out["text_features"],
                            logit_scale=model_out["logit_scale"],
                            logit_bias=model_out["logit_bias"],
                        )
                        val_loss = sum(losses.values())

                    # Statistics
                    running_val_loss.update(val_loss.detach())

                    # Update progress bar
                    progress.update(n=batch_size * args.world_size)

            time_now = time.time()
            rate = len(validation_dataset) / (time_now - epoch_start)
            with training_utils.single_handler_logging(logger, file_handler, enabled=not disable_tqdm) as log:
                log.info(
                    f"[Val] Epoch {epoch}/{epochs-1} "
                    f"Elapsed: {format_duration(time_now-epoch_start)}  "
                    f"R: {rate:.1f} samples/s"
                )

            progress.close()

            running_val_loss.synchronize_between_processes(device)
            epoch_val_loss = running_val_loss.global_avg

            # Write statistics
            if training_utils.is_global_primary(args) is True:
                summary_writer.add_scalars("loss", {"validation": epoch_val_loss}, epoch * epoch_samples)

            # Epoch validation metrics
            logger.info(f"[Val] Epoch {epoch}/{epochs-1} validation_loss: {epoch_val_loss:.4f}")

        # Learning rate scheduler update
        if step_update is False:
            scheduler.step()
        if last_lr != float(max(scheduler.get_last_lr())):
            last_lr = float(max(scheduler.get_last_lr()))
            logger.info(f"Updated learning rate to: {last_lr}")

        # Checkpoint model
        if epoch % args.save_frequency == 0:
            clip_training_utils.save_training_checkpoint(
                args,
                network_name,
                epoch,
                model_to_save,
                signature,
                rgb_stats,
                optimizer,
                scheduler,
                scaler,
                model_base,
            )
            if args.keep_last is not None and training_utils.is_global_primary(args) is True:
                fs_ops.clean_checkpoints(network_name, args.keep_last)

        # Epoch timing
        toc = time.time()
        logger.info(f"Total time: {format_duration(toc - tic)}")
        logger.info("---")

    # Save model hyperparameters with metrics
    if training_utils.is_global_primary(args) is True:
        # Replace list/dict based args
        if args.opt_betas is not None:
            for idx, beta in enumerate(args.opt_betas):
                setattr(args, f"opt_betas_{idx}", beta)

            del args.opt_betas

        if args.lr_steps is not None:
            args.lr_steps = json.dumps(args.lr_steps)
        if args.model_config is not None:
            args.model_config = json.dumps(args.model_config)
        if args.image_encoder_config is not None:
            args.image_encoder_config = json.dumps(args.image_encoder_config)
        if args.text_encoder_config is not None:
            args.text_encoder_config = json.dumps(args.text_encoder_config)
        if args.size is not None:
            args.size = json.dumps(args.size)

        metrics = {"hparam/loss": running_loss.global_avg}
        if running_val_loss.count > 0:
            metrics["hparam/val_loss"] = running_val_loss.global_avg

        # Save all args
        summary_writer.add_hparams({**vars(args), "training_samples": len(training_dataset)}, metrics)

    summary_writer.close()

    # Checkpoint model
    if epoch >= begin_epoch:
        clip_training_utils.save_training_checkpoint(
            args,
            network_name,
            epoch,
            model_to_save,
            signature,
            rgb_stats,
            optimizer,
            scheduler,
            scaler,
            model_base,
        )

    training_utils.shutdown_distributed_mode(args)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Train image-text model",
        epilog=(
            "Usage examples\n"
            "==============\n"
            "Training from an image-text dataset:\n"
            "python -m birder_clip.scripts.train \\\n"
            "    --network clip \\\n"
            "    --image-encoder vit_b16 \\\n"
            "    --text-encoder text_transformer \\\n"
            "    --embed-dim 512 \\\n"
            "    --tokenizer openai_clip_bpe \\\n"
            "    --loss clip \\\n"
            "    --batch-size 128 \\\n"
            "    --opt adamw --opt-fused \\\n"
            "    --lr 0.0005 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --epochs 200 --warmup-epochs 20 \\\n"
            "    --use-grayscale --resize-min-scale 0.4 \\\n"
            "    --rgb-mode clip \\\n"
            "    --amp --amp-dtype bfloat16 \\\n"
            "    --compile \\\n"
            "    --data-path data/training.csv \\\n"
            "    --val-path data/validation.csv\n"
            "\n"
            "python -m birder_clip.scripts.train \\\n"
            "    --network clip \\\n"
            "    --image-encoder vit_b16_avg \\\n"
            "    --text-encoder text_transformer \\\n"
            "    --embed-dim 512 \\\n"
            "    --loss clip \\\n"
            "    --batch-size 384 \\\n"
            "    --opt adamw --opt-fused \\\n"
            "    --lr 0.0005 \\\n"
            "    --lr-scheduler cosine \\\n"
            "    --epochs 100 --warmup-epochs 4 \\\n"
            "    --size 192 --context-length 32 \\\n"
            "    --use-grayscale --resize-min-scale 0.4 \\\n"
            "    --rgb-mode clip \\\n"
            "    --amp --amp-dtype bfloat16 \\\n"
            "    --compile \\\n"
            "    --wds \\\n"
            "    --wds-info https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/_info.json \\\n"
            "    --wds-split train\n"
        ),
        formatter_class=cli.ArgumentHelpFormatter,
    )
    training_cli.add_model_args(parser)
    training_cli.add_loss_args(parser)
    training_cli.add_optimization_args(parser)
    training_cli.add_lr_wd_args(parser)
    training_cli.add_lr_scheduler_args(parser)
    training_cli.add_training_schedule_args(parser)
    training_cli.add_ema_args(parser)
    training_cli.add_batch_norm_args(parser)
    training_cli.add_input_args(parser)
    training_cli.add_data_aug_args(parser)
    training_cli.add_dataloader_args(parser)
    training_cli.add_precision_args(parser)
    training_cli.add_compile_args(parser)
    training_cli.add_checkpoint_args(parser)
    training_cli.add_distributed_args(parser)
    training_cli.add_logging_and_debug_args(parser)
    training_cli.add_training_data_args(parser)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    # NOTE: Top-level model options are controlled by dedicated CLI arguments, not by --model-config

    args.size = cli.parse_size(args.size)
    training_cli.common_args_validation(args)

    reserved_model_config_keys = lib.get_reserved_model_config_keys(args.model_config)
    if len(reserved_model_config_keys) > 0:
        raise cli.ValidationError(
            "--model-config cannot contain keys with dedicated CLI flags: " f"{', '.join(reserved_model_config_keys)}"
        )
    if args.image_encoder_config is not None and "size" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain size, use --size")
    if args.image_encoder_config is not None and "input_channels" in args.image_encoder_config:
        raise cli.ValidationError("--image-encoder-config cannot contain input_channels, use --channels")
    if args.text_encoder_config is not None and "context_length" in args.text_encoder_config:
        raise cli.ValidationError("--text-encoder-config cannot contain context_length, use --context-length")

    if args.wds is True and args.data_path is None and args.wds_info is None:
        raise cli.ValidationError("Must provide at least one data source, --data-path or --wds-info")
    if args.wds is True and args.wds_info is not None and args.val_path is not None:
        raise cli.ValidationError("--val-path cannot be used with --wds-info")
    if args.rgb_mean is not None and len(args.rgb_mean) != args.channels:
        raise cli.ValidationError(f"--rgb-mean must have {args.channels} values, got {len(args.rgb_mean)}")
    if args.rgb_std is not None and len(args.rgb_std) != args.channels:
        raise cli.ValidationError(f"--rgb-std must have {args.channels} values, got {len(args.rgb_std)}")
    if args.resize_min_scale is not None and (args.resize_min_scale <= 0.0 or args.resize_min_scale >= 1.0):
        raise cli.ValidationError(f"--resize-min-scale must be in range of (0, 1.0), got {args.resize_min_scale}")
    if args.center_crop > 1 or args.center_crop <= 0.0:
        raise cli.ValidationError(f"--center-crop must be in range of (0, 1.0], got {args.center_crop}")


def args_from_dict(**kwargs: Any) -> argparse.Namespace:
    parser = get_args_parser()
    parser.set_defaults(**kwargs)
    args = parser.parse_args([])
    validate_args(args)

    return args


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.wds_cache_dir is not None and Path(args.wds_cache_dir).exists() is False:
        logger.info(f"Creating {args.wds_cache_dir} directory...")
        Path(args.wds_cache_dir).mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    logger = logging.getLogger(getattr(__spec__, "name", __name__))
    main()
