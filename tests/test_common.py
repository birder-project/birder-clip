import unittest

from birder_clip.common import lib
from birder_clip.net.base import SignatureType


class TestLib(unittest.TestCase):
    def test_signature_helpers(self) -> None:
        signature: SignatureType = {
            "inputs": [
                {"data_shape": [0, 3, 224, 256]},
                {"data_shape": [0, 77]},
            ],
            "outputs": [{"data_shape": [0, 0]}],
        }

        self.assertEqual(lib.get_size_from_signature(signature), (224, 256))
        self.assertEqual(lib.get_channels_from_signature(signature), 3)
        self.assertEqual(lib.get_context_length_from_signature(signature), 77)

    def test_image_text_network_name(self) -> None:
        self.assertEqual(
            lib.get_image_text_network_name(
                "siglip",
                image_encoder="maxvit_s",
                text_encoder="siglip2_text_b",
                embed_dim=512,
            ),
            "siglip_maxvit_s_siglip2_text_b_d512",
        )
        self.assertEqual(
            lib.get_image_text_network_name(
                "clip",
                image_encoder="maxvit_s",
                text_encoder="transformer_encoder",
                embed_dim=512,
                tokenizer="openai_clip_bpe",
                tag="exp",
            ),
            "clip_maxvit_s_openai_clip_bpe_d512_exp",
        )
        self.assertEqual(
            lib.get_image_text_network_name(
                "openvision_v2",
                image_encoder="vit_l14",
                text_decoder="visual_causal_text_decoder_s",
            ),
            "openvision_v2_vit_l14_visual_causal_text_decoder_s",
        )

    def test_image_text_model_config(self) -> None:
        base_config = {
            "image": {
                "network": "vit_b16",
                "config": {
                    "drop_path_rate": 0.1,
                },
                "size": (224, 224),
            },
            "text": {
                "network": "transformer_encoder",
                "config": {
                    "hidden_dim": 512,
                },
                "context_length": 77,
            },
            "embed_dim": 512,
            "tokenizer": "simple_tokenizer",
        }

        model_config = lib.get_image_text_model_config(
            base_config,
            {"image": {"extra": True}, "projection_dropout": 0.2},
            image_encoder="vit_l14",
            text_encoder_config={"num_layers": 12},
            input_channels=1,
            image_size=(336, 336),
            context_length=64,
            tokenizer="openai_clip_bpe",
        )

        assert model_config is not None
        self.assertEqual(model_config["image"]["network"], "vit_l14")
        self.assertEqual(model_config["image"]["input_channels"], 1)
        self.assertEqual(model_config["image"]["size"], (336, 336))
        self.assertEqual(model_config["image"]["extra"], True)
        self.assertEqual(model_config["image"]["config"]["drop_path_rate"], 0.1)
        self.assertEqual(model_config["text"]["network"], "transformer_encoder")
        self.assertEqual(model_config["text"]["config"]["hidden_dim"], 512)
        self.assertEqual(model_config["text"]["config"]["num_layers"], 12)
        self.assertEqual(model_config["text"]["context_length"], 64)
        self.assertEqual(model_config["embed_dim"], 512)
        self.assertEqual(model_config["tokenizer"], "openai_clip_bpe")
        self.assertEqual(model_config["projection_dropout"], 0.2)
        self.assertEqual(base_config["image"]["network"], "vit_b16")  # type: ignore[index]

    def test_image_text_model_config_empty(self) -> None:
        self.assertIsNone(lib.get_image_text_model_config())

    def test_image_text_model_config_decoder_only_context_length(self) -> None:
        model_config = lib.get_image_text_model_config(
            {
                "image": {"network": "vit_s16"},
                "decoder": {
                    "network": "visual_causal_text_decoder",
                    "context_length": 128,
                },
                "tokenizer": "openvision_v2",
            },
            context_length=64,
        )

        assert model_config is not None
        self.assertNotIn("text", model_config)
        self.assertEqual(model_config["decoder"]["context_length"], 64)
