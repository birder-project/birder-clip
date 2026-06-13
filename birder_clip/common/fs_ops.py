import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import NamedTuple
from typing import Optional

import torch
from birder.common import cli
from birder.conf import settings
from birder.data.transforms.classification import RGBType
from birder.data.transforms.classification import inference_preset

from birder_clip.common import lib
from birder_clip.model_registry import Task
from birder_clip.model_registry import registry
from birder_clip.model_registry.manifest import EncoderMetadataType
from birder_clip.model_registry.manifest import FileFormatType
from birder_clip.net.base import BaseNet
from birder_clip.net.base import SignatureType
from birder_clip.tokenizers import Tokenizer
from birder_clip.tokenizers import get_tokenizer
from birder_clip.tokenizers.hf import download_hf_tokenizer
from birder_clip.tokenizers.hf import get_hf_tokenizer_source
from birder_clip.version import __version__

try:
    import safetensors
    import safetensors.torch

    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False

logger = logging.getLogger(__name__)


class ModelInfo(NamedTuple):
    signature: SignatureType
    rgb_stats: RGBType
    custom_config: Optional[dict[str, Any]] = None


def write_config(network_name: str, net: BaseNet, signature: SignatureType, rgb_stats: RGBType) -> None:
    model_config = lib.get_image_text_network_config(net, signature, rgb_stats)
    config_file = settings.MODELS_DIR.joinpath(f"{network_name}.json")
    logger.info(f"Writing {config_file}")
    with open(config_file, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)


def _split_encoder_metadata(encoder: Optional[EncoderMetadataType]) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if encoder is None:
        return (None, None)
    if isinstance(encoder, str):
        return (encoder, None)

    if "network" not in encoder:
        raise ValueError("Encoder metadata must include a 'network' field")

    return (None, encoder)  # type: ignore[return-value]


def model_path(
    network_name: str,
    *,
    epoch: Optional[int | str] = None,
    st: bool = False,
    states: bool = False,
) -> Path:
    if epoch is not None:
        file_name = f"{network_name}_{epoch}"
    else:
        file_name = network_name

    if states is True:
        file_name = f"{file_name}_states.pt"
    elif st is True:
        file_name = f"{file_name}.safetensors"
    else:
        file_name = f"{file_name}.pt"

    return settings.MODELS_DIR.joinpath(file_name)


def _checkpoint_states(
    states_path: Path,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    model_base: Optional[torch.nn.Module],
    **extra_states: Optional[dict[str, Any]],
) -> None:
    if optimizer is None and scheduler is None and scaler is None and model_base is None and len(extra_states) == 0:
        return

    kwargs = {}
    if optimizer is not None:
        kwargs["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        kwargs["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        kwargs["scaler_state"] = scaler.state_dict()
    if model_base is not None:
        kwargs["model_base_state"] = model_base.state_dict()
    kwargs.update({k: v for k, v in extra_states.items() if v is not None})

    logger.info(f"Saving training states {states_path}...")
    torch.save(kwargs, states_path)


def _checkpoint_states_from_state_dicts(
    states_path: Path,
    optimizer_state: Optional[dict[str, Any]],
    scheduler_state: Optional[dict[str, Any]],
    scaler_state: Optional[dict[str, Any]],
    model_base_state: Optional[dict[str, Any]],
    **extra_states: Optional[dict[str, Any]],
) -> None:
    if optimizer_state is None or scheduler_state is None:
        return

    logger.info(f"Saving checkpoint states {states_path}...")
    torch.save(
        {
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
            "model_base_state": model_base_state,
            **extra_states,
        },
        states_path,
    )


class TrainingStates(NamedTuple):
    optimizer_state: Optional[dict[str, Any]]
    scheduler_state: Optional[dict[str, Any]]
    scaler_state: Optional[dict[str, Any]]
    model_base_state: Optional[dict[str, Any]]
    ema_model_state: Optional[dict[str, Any]] = None
    extra_states: Optional[dict[str, Any]] = None

    @classmethod
    def empty(cls) -> "TrainingStates":
        return cls(None, None, None, None, None)


def _load_states(states_path: Path, device: torch.device) -> TrainingStates:
    if states_path.exists() is True:
        logger.info(f"Loading states from {states_path} on device {device}...")
        states_dict: dict[str, Any] = torch.load(states_path, map_location=device, weights_only=True)
        optimizer_state = states_dict.pop("optimizer_state", None)
        scheduler_state = states_dict.pop("scheduler_state", None)
        scaler_state = states_dict.pop("scaler_state", None)
        model_base_state = states_dict.pop("model_base_state", None)
        extra_states = {}
        for state in states_dict:
            extra_states[state] = states_dict[state]

        return TrainingStates(
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            scaler_state=scaler_state,
            model_base_state=model_base_state,
            extra_states=extra_states,
        )

    logger.debug("Checkpoint training states not found, returning empty states")
    return TrainingStates.empty()


def checkpoint_model(
    network_name: str,
    epoch: int,
    net: torch.nn.Module,
    signature: SignatureType,
    rgb_stats: RGBType,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.grad_scaler.GradScaler],
    model_base: Optional[torch.nn.Module],
    *,
    external_config: Optional[dict[str, Any]] = None,
    **extra_states: Optional[dict[str, Any]],
) -> None:
    kwargs = {}
    if external_config is not None:
        kwargs["config"] = external_config

    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)
    logger.info(f"Saving model checkpoint {path}...")
    torch.save(
        {
            "state": net.state_dict(),
            "birder_clip_version": __version__,
            "task": net.task,
            "signature": signature,
            "rgb_stats": rgb_stats,
            **kwargs,
        },
        path,
    )

    _checkpoint_states(states_path, optimizer, scheduler, scaler, model_base, **extra_states)


def checkpoint_model_from_state_dicts(
    network_name: str,
    epoch: int,
    model_state: dict[str, Any],
    task: Any,
    signature: SignatureType,
    rgb_stats: RGBType,
    optimizer_state: Optional[dict[str, Any]],
    scheduler_state: Optional[dict[str, Any]],
    scaler_state: Optional[dict[str, Any]],
    model_base_state: Optional[dict[str, Any]],
    *,
    external_config: Optional[dict[str, Any]] = None,
    **extra_states: Optional[dict[str, Any]],
) -> None:
    kwargs = {}
    if external_config is not None:
        kwargs["config"] = external_config

    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)
    logger.info(f"Saving model checkpoint {path}...")
    torch.save(
        {
            "state": model_state,
            "birder_clip_version": __version__,
            "task": task,
            "signature": signature,
            "rgb_stats": rgb_stats,
            **kwargs,
        },
        path,
    )

    _checkpoint_states_from_state_dicts(
        states_path,
        optimizer_state,
        scheduler_state,
        scaler_state,
        model_base_state,
        **extra_states,
    )


def clean_checkpoints(network_name: str, keep_last: int) -> None:
    epoch = "*[0-9]"
    models_glob = str(model_path(network_name, epoch=epoch))
    states_glob = str(model_path(network_name, epoch=epoch, states=True))
    model_pattern = re.compile(r".*_([1-9][0-9]*)\.pt$")
    states_pattern = re.compile(r".*_([1-9][0-9]*)_states\.pt$")

    model_paths = list(settings.BASE_DIR.glob(models_glob))
    for p in sorted(model_paths, key=lambda p: p.stat().st_mtime)[:-keep_last]:
        if model_pattern.search(str(p)) is not None:
            logger.info(f"Removing checkpoint {p}...")
            p.unlink()

    state_paths = list(settings.BASE_DIR.glob(states_glob))
    for p in sorted(state_paths, key=lambda p: p.stat().st_mtime)[:-keep_last]:
        if states_pattern.search(str(p)) is not None:
            logger.info(f"Removing checkpoint states {p}...")
            p.unlink()


class CheckpointStates(NamedTuple):
    net: BaseNet
    rgb_stats: RGBType
    training_states: TrainingStates


def load_checkpoint(
    device: torch.device,
    network: str,
    *,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    image_encoder: Optional[str] = None,
    text_encoder: Optional[str] = None,
    embed_dim: Optional[int] = None,
    tokenizer: Optional[str] = None,
    image_encoder_config: Optional[dict[str, Any]] = None,
    text_encoder_config: Optional[dict[str, Any]] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
    new_context_length: Optional[int] = None,
    strict: bool = True,
) -> CheckpointStates:
    network_name = lib.get_image_text_network_name(
        network,
        tag=tag,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        tokenizer=tokenizer,
    )
    path = model_path(network_name, epoch=epoch)
    states_path = model_path(network_name, epoch=epoch, states=True)

    logger.info(f"Loading model from {path} on device {device}...")
    model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    training_states = _load_states(states_path, device)

    signature: SignatureType = model_dict["signature"]
    rgb_stats: RGBType = model_dict["rgb_stats"]
    input_channels = lib.get_channels_from_signature(signature)
    size = lib.get_size_from_signature(signature)
    context_length = lib.get_context_length_from_signature(signature)
    logger.debug(f"Loaded model with RGB stats: {rgb_stats}")
    logger.debug(f"Loaded model input size is {size}")

    registered_config = registry.all_nets[network.lower()].config  # type: ignore[misc]
    loaded_config = model_dict.get("config", {})
    checkpoint_config = {} if loaded_config is None else loaded_config.copy()
    if config is not None:
        checkpoint_config.update(config)

    model_config = lib.get_image_text_model_config(
        registered_config,
        checkpoint_config,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        tokenizer=tokenizer,
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        input_channels=input_channels,
        image_size=size,
        context_length=context_length,
    )
    net = registry.net_factory(network, config=model_config)

    if training_states.model_base_state is not None:
        net.load_state_dict(training_states.model_base_state, strict=strict)
        training_states = training_states._replace(ema_model_state=model_dict["state"])
    else:
        net.load_state_dict(model_dict["state"], strict=strict)

    if new_size is not None:
        net.adjust_image_size(new_size)
    if new_context_length is not None:
        net.adjust_context_length(new_context_length)

    net.to(device)

    return CheckpointStates(net, rgb_stats, training_states)


def load_model(
    device: torch.device,
    network: str,
    *,
    path: Optional[str | Path] = None,
    config: Optional[dict[str, Any]] = None,
    tag: Optional[str] = None,
    image_encoder: Optional[str] = None,
    text_encoder: Optional[str] = None,
    embed_dim: Optional[int] = None,
    tokenizer: Optional[str] = None,
    image_encoder_config: Optional[dict[str, Any]] = None,
    text_encoder_config: Optional[dict[str, Any]] = None,
    epoch: Optional[int] = None,
    new_size: Optional[tuple[int, int]] = None,
    new_context_length: Optional[int] = None,
    inference: bool,
    st: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> tuple[BaseNet, ModelInfo]:
    if path is None:
        _network_name = lib.get_image_text_network_name(
            network,
            tag=tag,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            tokenizer=tokenizer,
        )
        path = model_path(_network_name, epoch=epoch, st=st)

    logger.info(f"Loading model from {path} on device {device}...")

    if st is True:
        assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
        with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
            extra_files = handle.metadata()
        assert extra_files is not None

        signature: SignatureType = json.loads(extra_files["signature"])
        rgb_stats: RGBType = json.loads(extra_files["rgb_stats"])
        if "config" in extra_files and len(extra_files["config"]) > 0:
            loaded_config: dict[str, Any] = json.loads(extra_files["config"])
        else:
            loaded_config = {}

    else:
        model_dict: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
        signature = model_dict["signature"]
        rgb_stats = model_dict["rgb_stats"]
        loaded_config = model_dict.get("config", {})

    size = lib.get_size_from_signature(signature)
    input_channels = lib.get_channels_from_signature(signature)
    context_length = lib.get_context_length_from_signature(signature)
    registered_config = registry.all_nets[network.lower()].config  # type: ignore[misc]
    checkpoint_config = {} if loaded_config is None else loaded_config.copy()
    if config is not None:
        checkpoint_config.update(config)

    model_config = lib.get_image_text_model_config(
        registered_config,
        checkpoint_config,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=embed_dim,
        tokenizer=tokenizer,
        image_encoder_config=image_encoder_config,
        text_encoder_config=text_encoder_config,
        input_channels=input_channels,
        image_size=size,
        context_length=context_length,
    )

    net = registry.net_factory(network, config=model_config)
    if st is True:
        model_state: dict[str, Any] = safetensors.torch.load_file(path, device=device.type)
        net.load_state_dict(model_state)
    else:
        net.load_state_dict(model_dict["state"])

    if new_size is not None:
        net.adjust_image_size(new_size)
    if new_context_length is not None:
        net.adjust_context_length(new_context_length)

    net.to(device)
    if dtype is not None:
        net.to(dtype)
    if inference is True:
        for param in net.parameters():
            param.requires_grad_(False)

        net.eval()

    if len(loaded_config) == 0:
        custom_config = None
    else:
        custom_config = loaded_config
        logger.debug(f"Model loaded with custom config: {custom_config}")

    return (net, ModelInfo(signature, rgb_stats, custom_config))


def load_pretrained_model(
    weights: str,
    *,
    dst: Optional[str | Path] = None,
    file_format: FileFormatType = "pt",
    inference: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    custom_config: Optional[dict[str, Any]] = None,
    progress_bar: bool = True,
) -> tuple[BaseNet, ModelInfo]:
    """
    Load a pretrained model

    Parameters
    ----------
    weights
        Name of the pretrained weights to load from the model registry.
    dst
        Destination path where the model weights will be downloaded or loaded from.
        If None, the model will be saved in the default models directory.
    file_format
        Model format.
    inference
        Whether to prepare the model for inference mode.
    device
        Device to load the model on.
    dtype
        Data type for model parameters and computations.
    custom_config
        Additional model configuration that overrides or extends the predefined configuration.
    progress_bar
        Whether to display a progress bar during file download.

    Returns
    -------
    A tuple containing two elements:
    - A PyTorch module (neural network model) loaded with pretrained weights.
    - Model info containing signature and RGB stats.

    Notes
    -----
    - Creates the models directory if it does not exist.
    - Downloads the model weights if not already present locally.
    - When inference is True, the model is set to evaluation mode with gradient calculation disabled.
    - If device is None, it will default to CPU.

    Examples
    --------
    >>> net, model_info = load_pretrained_model("openai_clip_vit_l14")
    >>> net, model_info = load_pretrained_model(
    ...     "openai_clip_vit_l14", inference=True, device=torch.device("cuda"))
    """

    download_model_by_weights(weights, dst=dst, file_format=file_format, progress_bar=progress_bar)
    model_metadata = registry.get_pretrained_metadata(weights)
    model_file, _ = lib.get_pretrained_model_url(weights, file_format)
    if dst is None:
        dst = settings.MODELS_DIR.joinpath(model_file)

    if device is None:
        device = torch.device("cpu")

    if model_metadata["task"] != Task.IMAGE_TEXT:
        raise ValueError(f"Unknown model type: {model_metadata['task']}")

    image_encoder, image_config = _split_encoder_metadata(model_metadata["net"].get("image_encoder", None))
    text_encoder, text_config = _split_encoder_metadata(model_metadata["net"].get("text_encoder", None))

    pretrained_config: dict[str, Any] = {}
    if image_config is not None:
        pretrained_config["image"] = image_config

    if text_config is not None:
        pretrained_config["text"] = text_config

    if custom_config is not None:
        pretrained_config.update(custom_config)
    if len(pretrained_config) > 0:
        config = pretrained_config
    else:
        config = None

    return load_model(
        device,
        model_metadata["net"]["network"],
        path=dst,
        config=config,
        tag=model_metadata["net"].get("tag", None),
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        embed_dim=model_metadata["net"].get("embed_dim", None),
        tokenizer=model_metadata["net"].get("tokenizer", None),
        inference=inference,
        st=file_format == "safetensors",
        dtype=dtype,
    )


def load_pretrained_model_and_transform(
    weights: str,
    *,
    dst: Optional[str | Path] = None,
    file_format: FileFormatType = "pt",
    inference: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    custom_config: Optional[dict[str, Any]] = None,
    progress_bar: bool = True,
    classification_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[BaseNet, ModelInfo, Callable[..., torch.Tensor]]:
    """
    Load a pretrained model and build the matching inference transform

    This is a convenience helper for the common inference path where the model and
    its default preprocessing are needed together. Image-text models use inference_preset.

    Parameters
    ----------
    weights
        Name of the pretrained weights to load from the model registry.
    dst
        Destination path where the model weights will be downloaded or loaded from.
    file_format
        Model format.
    inference
        Whether to prepare the model for inference mode.
    device
        Device to load the model on.
    dtype
        Data type for model parameters and computations.
    custom_config
        Additional model configuration that overrides or extends the predefined configuration.
    progress_bar
        Whether to display a progress bar during file download.
    classification_kwargs
        Optional keyword arguments forwarded to inference_preset.

    Returns
    -------
    A tuple containing three elements:
    - A PyTorch module (neural network model) loaded with pretrained weights.
    - Model info containing signature and RGB stats.
    - An inference transform matching the model task.
    """

    net, model_info = load_pretrained_model(
        weights,
        dst=dst,
        file_format=file_format,
        inference=inference,
        device=device,
        dtype=dtype,
        custom_config=custom_config,
        progress_bar=progress_bar,
    )

    size = lib.get_size_from_signature(model_info.signature)
    classification_args = {} if classification_kwargs is None else dict(classification_kwargs)
    transform = inference_preset(size, model_info.rgb_stats, **classification_args)

    return (net, model_info, transform)


def load_pretrained_tokenizer(weights: str, *, download: bool = True, **kwargs: Any) -> Tokenizer:
    """
    Load the tokenizer matching pretrained weights

    Parameters
    ----------
    weights
        Name of the pretrained weights to load from the model registry.
    download
        Whether to download tokenizer files when needed.
    kwargs
        Additional tokenizer keyword arguments that override or extend the predefined configuration.

    Returns
    -------
    A tokenizer configured for the pretrained weights.
    """

    model_metadata = registry.get_pretrained_metadata(weights)
    tokenizer_name = model_metadata["net"].get("tokenizer", None)
    if tokenizer_name is None:
        tokenizer_name = registry.get_default_tokenizer(model_metadata["net"]["network"])
    if tokenizer_name is None:
        raise ValueError(f"Tokenizer is not available for {weights}")

    tokenizer_kwargs = {"context_length": model_metadata["context_length"], **kwargs}
    if download is True:
        hf_source = get_hf_tokenizer_source(tokenizer_name)
        if hf_source is not None:
            # Match the source that get_tokenizer will use after applying caller overrides
            hf_source = tokenizer_kwargs.get("source", hf_source)
            download_hf_tokenizer(hf_source)

    return get_tokenizer(tokenizer_name, **tokenizer_kwargs)


def save_st(
    net: torch.nn.Module,
    dst: str,
    task: str,
    signature: SignatureType,
    rgb_stats: RGBType,
    *,
    external_config: Optional[dict[str, Any]] = None,
) -> None:
    assert _HAS_SAFETENSORS, "'pip install safetensors' to use .safetensors"
    kwargs = {}
    if external_config is not None:
        kwargs["config"] = json.dumps(external_config)

    safetensors.torch.save_model(
        net,
        str(dst),
        {
            "birder_clip_version": __version__,
            "task": task,
            "signature": json.dumps(signature),
            "rgb_stats": json.dumps(rgb_stats),
            **kwargs,
        },
    )


def download_model_by_weights(
    weights: str, *, dst: Optional[str | Path] = None, file_format: FileFormatType = "pt", progress_bar: bool = True
) -> None:
    if settings.MODELS_DIR.exists() is False:
        logger.info(f"Creating {settings.MODELS_DIR} directory...")
        settings.MODELS_DIR.mkdir(parents=True)

    model_metadata = registry.get_pretrained_metadata(weights)
    if file_format not in model_metadata["formats"]:
        available_formats = ", ".join(model_metadata["formats"].keys())
        raise ValueError(
            f"Requested format '{file_format}' not available for {weights}, available formats are: {available_formats}"
        )

    model_file, url = lib.get_pretrained_model_url(weights, file_format)
    if dst is None:
        dst = settings.MODELS_DIR.joinpath(model_file)

    cli.download_file(url, dst, model_metadata["formats"][file_format]["sha256"], progress_bar=progress_bar)
