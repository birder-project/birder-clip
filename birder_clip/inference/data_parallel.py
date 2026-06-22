"""
Inference-optimized multi-GPU parallelization for image-text models

This module provides ZeroShotInferenceDataParallel, a CLIP-style zero-shot
specialization of Birder's InferenceDataParallel.
"""

from typing import Optional

import torch
from birder.inference.data_parallel import InferenceDataParallel

from birder_clip.inference.zero_shot import ZeroShotInference
from birder_clip.net.base import BaseNet


class ZeroShotInferenceDataParallel(InferenceDataParallel):
    """
    Distributes zero-shot image inference batches across multiple GPUs

    This wrapper scatters the image batch across devices and keeps a replicated
    copy of the zero-shot text embeddings on each device. Each replica computes
    image embeddings and zero-shot logits locally before outputs are gathered.

    Important
    ---------
    This class assumes the model is already configured for inference mode
    (i.e., loaded with inference=True in load_model or manually set to eval mode
    with requires_grad=False on all parameters).
    """

    def __init__(
        self,
        module: BaseNet,
        text_embeddings: torch.Tensor,
        device_ids: Optional[list[int]] = None,
        output_device: Optional[int | str | torch.device] = None,
        compile_replicas: bool = False,
        compile_methods: Optional[list[str]] = None,
        compile_mode: Optional[str] = None,
    ) -> None:
        if compile_methods is None:
            compile_methods = ["encode_image", "forward_logits"]

        super().__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            compile_replicas=compile_replicas,
            compile_methods=compile_methods,
            compile_mode=compile_mode,
        )
        self.set_text_embeddings(text_embeddings)

    def set_text_embeddings(self, text_embeddings: torch.Tensor) -> None:
        if text_embeddings.ndim != 2:
            raise ValueError(
                f"text_embeddings must be a 2D tensor of shape (num_classes, embedding_size), "
                f"got shape {text_embeddings.size()}"
            )

        self.text_embeddings = [
            text_embeddings.to(f"cuda:{device_id}", non_blocking=True) for device_id in self.device_ids
        ]
        self.inference_modules = [
            ZeroShotInference(replica, embeddings) for replica, embeddings in zip(self.replicas, self.text_embeddings)
        ]

    def forward(  # type: ignore[override] # pylint: disable=arguments-differ
        self, inputs: torch.Tensor, *, tta: bool = False, return_logits: bool = False
    ) -> torch.Tensor:
        """
        Run zero-shot inference distributed across GPUs

        Parameters
        ----------
        inputs
            Input image batch to process.
        tta
            Run inference with oversampling.
        return_logits
            If True, return raw logits instead of probabilities after softmax.
        """

        if len(self.device_ids) == 1:
            output = self.inference_modules[0](
                inputs,
                tta=tta,
                return_logits=return_logits,
            )
            return self._gather([output])

        scattered = self._scatter(inputs, {})

        outputs = []
        for inference, (input_chunk, _), device_id in zip(self.inference_modules, scattered, self.device_ids):
            if input_chunk is not None and input_chunk.size(0) > 0:
                with torch.cuda.device(device_id):
                    output = inference(
                        input_chunk,
                        tta=tta,
                        return_logits=return_logits,
                    )
                    outputs.append(output)
            else:
                outputs.append(None)

        return self._gather(outputs)

    def __repr__(self) -> str:
        return (
            f"ZeroShotInferenceDataParallel(\n"
            f"  devices={self.device_ids},\n"
            f"  output_device={self.output_device},\n"
            f"  src_device={self.src_device},\n"
            f"  text_embeddings_shape={tuple(self.text_embeddings[0].shape)}\n"
            f")"
        )
