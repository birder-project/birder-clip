# Reference Training Procedure

## Network Specific Training Procedures

- [CLIP](#clip)
- [CoCa](#coca)
- [OpenVision v2](#openvision-v2)
- [LiT](#lit)

### CLIP

#### CLIP: ViT b16, text transformer, d512

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network clip --image-encoder vit_b16 --text-encoder transformer_encoder --embed-dim 512 --loss clip --batch-size 384 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 192 --context-length 32 --aug-type clip --clip-color-jitter-prob 0.6 --clip-gray-prob 0.2 --rgb-mode clip --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```

With gradient checkpointing

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network clip --image-encoder vit_b16 --text-encoder transformer_encoder --embed-dim 512 --loss clip --batch-size 512 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 192 --context-length 32 --aug-type clip --clip-color-jitter-prob 0.6 --clip-gray-prob 0.2 --rgb-mode clip --amp --amp-dtype bfloat16 --grad-checkpointing --compile --data-path data/training.csv
```

With gradient accumulation

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network clip --image-encoder vit_b16 --text-encoder transformer_encoder --embed-dim 512 --loss clip --batch-size 384 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --grad-accum-steps 16 --grad-accum-cache-negatives --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 192 --context-length 32 --aug-type clip --clip-color-jitter-prob 0.6 --clip-gray-prob 0.2 --rgb-mode clip --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```

### CoCa

#### CoCa: ViT b16, text transformer, d512

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network coca --image-encoder vit_b16 --text-encoder transformer_encoder --embed-dim 512 --loss coca --coca-caption-loss-weight 2.0 --batch-size 256 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 192 --context-length 32 --aug-type clip --clip-gray-prob 0.2 --rgb-mode clip --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```

For caption-only fine-tuning from a pretrained CoCa checkpoint, disable the contrastive loss and use a lower learning rate, e.g. `--coca-contrastive-loss-weight 0.0 --lr 1e-5`.

### OpenVision v2

#### OpenVision v2: ViT l14

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network openvision_v2 --image-encoder vit_l14 --loss caption --batch-size 256 --opt adamw --opt-fused --opt-betas 0.9 0.95 --grad-accum-steps 8 --lr 0.001 --wd 0.2 --lr-scheduler cosine --lr-cosine-min 0.0 --epochs 800 --warmup-epochs 5 --size 112 --context-length 128 --aug-type clip --clip-gray-prob 0.2 --rgb-mode clip --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```

For the 224px fine-tune stage, switch to `--lr 2.5e-5` for about 15% of the total training length.

#### OpenVision v2: RoPE SoViT reg8 150m p14 swiglu rms AVG (frozen image encoder)

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network openvision_v2 --image-encoder rope_vit_reg8_so150m_p14_swiglu_rms_avg --image-encoder-pretrained models/rope_vit_reg8_so150m_p14_swiglu_rms_avg_capi.pt --loss caption --freeze-image-encoder --batch-size 256 --opt adamw --opt-fused --opt-betas 0.9 0.95 --grad-accum-steps 8 --lr 0.001 --wd 0.2 --lr-scheduler cosine --lr-cosine-min 0.0 --epochs 200 --warmup-epochs 5 --size 224 --context-length 128 --aug-type clip --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```

### LiT

#### LiT: Bio-DINO, text transformer, d512

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network clip --tag lit --image-encoder vit_reg4_so150m_p14_ls --image-encoder-pretrained models/vit_reg4_so150m_p14_ls_dino-v2-bio-224px --text-encoder transformer_encoder --embed-dim 512 --loss clip --freeze-image-encoder --batch-size 384 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 224 --context-length 48 --aug-type clip --rgb-mode centered --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```
