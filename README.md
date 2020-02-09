# Q-Layer
Code for Q-Layer: Quantization Layer for Robust Convolutional Neural Network

## To Use
### Prepare Data
Download data to ./data

### Generate validation set
`python rawtrain.py --task CIFAR_CNN_1 --gpu 1 --save_best 1 --early_stop 40 --epochs 100`


### Generate validation set
Train a target model, then do as follows (remember to change the path in the CIFAR_generate_black_box_data() function)

`python adv_test.py --task CIFAR_generate_black_box_data`

### Raw train
`python rawtrain.py --task CIFAR_VQ --gpu 1 --save_best 1 --early_stop 40 --epochs 100 --update train --set_fixed 4 --num_concept 64 --inter_layer vgg16_pool0.5`

### Adv train
`python adv_train.py --task CIFAR_VQ --gpu 1 --load_path ./save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1_1_0.001_0.001_lr_0.001_dim_16_share_True_update_train/  --eps 0.02 --early_stop 40 --save_best 1 --ratio 1 --epochs 100`

### Adv Test
`python adv_test.py --task CIFAR --gpu 0 --vq_path ./save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1.0_1_10.0_10.0_lr_0.001_dim_16_share_True_update_train/ --FGSM_black True --BIM_black True --FGSM_white True --BIM_white True --CIM_white True --eps [0.02,0.05]`


