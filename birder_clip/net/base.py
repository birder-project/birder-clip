import copy
from collections.abc import Callable
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import TypedDict

import torch
from birder.net.base import BaseNet as ImageEncoder
from torch import nn
from transformers import GenerationConfig
from transformers import PretrainedConfig
from transformers.generation.utils import GenerateOutput
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from birder_clip.model_registry import Task
from birder_clip.net.text.base import TextDecoderBaseNet
from birder_clip.net.text.base import TextEncoderBaseNet

DataShapeType = TypedDict("DataShapeType", {"data_shape": list[int]})
SignatureType = TypedDict("SignatureType", {"inputs": list[DataShapeType], "outputs": list[DataShapeType]})


def get_image_text_signature(
    input_shape: tuple[int, ...], context_length: int, *, output_shape: tuple[int, ...] = (0, 0)
) -> SignatureType:
    return {
        "inputs": [
            {"data_shape": [0, *input_shape[1:]]},
            {"data_shape": [0, context_length]},
        ],
        "outputs": [{"data_shape": list(output_shape)}],
    }


class BaseNet(nn.Module):
    task: ClassVar[str] = str(Task.IMAGE_TEXT)

    def __init__(self, *, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        if hasattr(self, "config") is False:
            self.config = config
        else:
            if self.config is not None:
                self.config = copy.deepcopy(self.config)  # Avoid mutating registered configs

        if config is not None:
            assert self.config is not None
            self.config.update(config)  # Override with custom config

        self.image_encoder: ImageEncoder  # CoCa assumes a detector backbone
        self.text_encoder: Optional[TextEncoderBaseNet] = None
        self.text_decoder: Optional[TextDecoderBaseNet] = None
        self.embedding_size: int
        self.tokenizer_name: str

    def get_signature_output_shape(self) -> tuple[int, ...]:
        return (0, 0)

    def set_grad_checkpointing(
        self,
        enable: bool = True,
        *,
        segments: Optional[int] = None,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        raise RuntimeError(f"{self.__class__.__name__} does not support gradient checkpointing")

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        *,
        return_features: bool = False,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError

    def adjust_image_size(self, new_size: tuple[int, int]) -> None:
        self.image_encoder.adjust_size(new_size)

    def adjust_context_length(self, new_context_length: int) -> None:
        if self.text_encoder is not None:
            self.text_encoder.adjust_context_length(new_context_length)
        if self.text_decoder is not None:
            self.text_decoder.adjust_context_length(new_context_length)


class ZeroShotMixin:
    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, text: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def forward_logits(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CaptionGeneratorMixin:
    text_decoder: TextDecoderBaseNet

    def forward_caption_features(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward_caption_logits(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        *,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def prepare_generation_input_ids(
        image: torch.Tensor, input_ids: Optional[torch.LongTensor], start_token_id: Optional[int]
    ) -> torch.LongTensor:
        if input_ids is not None and start_token_id is not None:
            raise ValueError("Caption generation accepts either input_ids or start_token_id, not both")

        if input_ids is None:
            if start_token_id is None:
                raise ValueError("Caption generation requires input_ids or start_token_id")

            input_ids = torch.full((image.size(0), 1), start_token_id, dtype=torch.long, device=image.device)

        if input_ids.size(0) != image.size(0):
            raise ValueError(
                "Caption generation requires image and input_ids batch sizes to match before generation: "
                f"image batch is {image.size(0)}, input_ids batch is {input_ids.size(0)}"
            )

        return input_ids

    def generate(
        self,
        image: torch.Tensor,
        input_ids: Optional[torch.LongTensor] = None,
        start_token_id: Optional[int] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs: Any,
    ) -> torch.LongTensor | GenerateOutput:
        raise NotImplementedError


def generate_with_hf(
    input_ids: torch.LongTensor,
    *,
    context_length: int,
    vocab_size: int,
    forward: Callable[..., torch.Tensor],
    generation_config: Optional[GenerationConfig] = None,
    **kwargs: Any,
) -> torch.LongTensor | GenerateOutput:
    if kwargs.get("attention_mask") is not None:
        raise ValueError("Caption generation does not support attention_mask")
    if kwargs.get("use_cache") is True:
        raise ValueError("Caption generation does not support use_cache=True")

    kwargs["use_cache"] = False
    model = _HFGenerationWrapper(
        context_length=context_length,
        vocab_size=vocab_size,
        device=input_ids.device,
        forward=forward,
    )

    return model.generate(input_ids, generation_config=generation_config, **kwargs)


class _CaptionGenerationConfig(PretrainedConfig):
    def __init__(self, *, vocab_size: int = 0, context_length: int = 0) -> None:
        super().__init__(is_encoder_decoder=False)
        self.vocab_size = vocab_size
        self.max_position_embeddings = context_length


class _HFGenerationWrapper(nn.Module, GenerationMixin):
    main_input_name: ClassVar[str] = "input_ids"

    def __init__(
        self, *, context_length: int, vocab_size: int, device: torch.device, forward: Callable[..., torch.Tensor]
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.generation_device = device
        self.forward_fn = forward
        self.config = _CaptionGenerationConfig(vocab_size=vocab_size, context_length=context_length)
        self.generation_config = GenerationConfig(use_cache=False)

    @property
    def device(self) -> torch.device:
        return self.generation_device

    def get_experts_implementation(self) -> dict[str, None]:
        # GenerationMixin expects this PreTrainedModel hook, but the wrapper has no MoE layers
        return {"": None}

    def _validate_generated_length(
        self, generation_config: GenerationConfig, input_ids_length: int, has_default_max_length: bool
    ) -> None:
        super()._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        if generation_config.max_length is not None and generation_config.max_length > self.context_length:
            raise ValueError(
                f"Generation max_length={generation_config.max_length} exceeds decoder context length "
                f"{self.context_length}"
            )

    def prepare_inputs_for_generation(  # type: ignore[override] # pylint: disable=arguments-differ
        self, input_ids: torch.LongTensor, **kwargs: Any
    ) -> dict[str, Any]:
        return {"input_ids": input_ids, **kwargs}

    def forward(  # pylint: disable=unused-argument
        self, input_ids: torch.LongTensor, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        logits = self.forward_fn(input_ids)
        return CausalLMOutputWithPast(logits=logits)
