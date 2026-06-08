# Reference Training Procedure

## Network Specific Training Procedures

- [CLIP](#clip)

### CLIP

#### CLIP: ViT b16, text transformer, d512

```sh
torchrun --nproc_per_node=2 -m birder_clip.scripts.train --network clip --image-encoder vit_b16 --text-encoder text_transformer --embed-dim 512 --loss clip --batch-size 384 --opt adamw --opt-fused --opt-eps 1e-6 --opt-betas 0.9 0.98 --lr 0.001 --lr-scheduler cosine --epochs 200 --warmup-epochs 5 --size 192 --context-length 32 --aug-level 4 --use-grayscale --resize-min-scale 0.4 --rgb-mode clip --amp --amp-dtype bfloat16 --compile --data-path data/training.csv
```
