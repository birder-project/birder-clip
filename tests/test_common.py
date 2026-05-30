import unittest

from birder_clip.common import lib


class TestLib(unittest.TestCase):
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
                text_encoder="text_transformer",
                embed_dim=512,
                tokenizer="openai_clip_bpe",
                tag="exp",
            ),
            "clip_maxvit_s_text_transformer_openai_clip_bpe_d512_exp",
        )
