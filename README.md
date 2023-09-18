# Correlation-Induced Label Prior for Semi-Supervised Multi-Label Learning

Code for the paper "Correlation-Induced Label Prior for Semi-Supervised Multi-Label Learning".

## Setting Data

See the `README.md` file in the `data` directory for instructions on downloading and setting up the datasets.

## Training scripts

### pascal

```bash
# lb_ratio 0.05
python main.py --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 10 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 20 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.1
python main.py --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 10 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
nohup python main.py --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 20 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.15
python main.py --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 10 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
nohup python main.py --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 20 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.2
python main.py --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 10 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 32 -p 50 --warmup_epochs 10 --epoch 20 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
```

### coco

```bash
# lb_ratio 0.05
python main.py --dataset_name coco --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name coco --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.1
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.15
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.2
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name coco --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
```

### nuswide

```bash
# lb_ratio 0.05
python main.py --dataset_name nus --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name nus --img_size 256 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.1
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.1 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.15
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.15 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
# lb_ratio 0.2
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 1 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --finetune false
python main.py --dataset_name nus --img_size 256 --lb_ratio 0.2 --lr_g 1e-4 --lr_e 1e-4 --lr_d 1e-4 --lr_p 1e-4 --lr_a 1e-4 --latent_dim 128 -p 50 --warmup_epochs 1 --epoch 30 -j 16 --sup_coef 1 --enc_coef 1e-3 --ub_coef 1e-1 --bs_ratio 8
```