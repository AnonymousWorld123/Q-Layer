# Q-Layer
Code for our paper "Q-Layer: Latent Space Contraints for Robust Convolutional Neural Network".

# To Use
The project is implemented with Tensorflow 1.15.
## Prepare Dataset
Download the MNIST/Fashion-MNIST/CIFAR-10 dataset to ./data/\[dataset name\]

### Protector trains a validation CNN
`python rawtrain.py --task MNIST_CNN_1 --fashion True --save_best 1 --early_stop 10 --epochs 40 --save_path cnn_validation/`

### Protector generates a validation set
Assign the source model path as the protector's validation CNN path. 
`python adv_test.py --task MNIST_generate_black_box_data --fashion True --cnn_path cnn_validation/ --eps [0.1] --validation_file True`

### Attacker trains a source CNN
`python rawtrain.py --task MNIST_CNN_1 --fashion True --save_best 1 --early_stop 10 --epochs 40 --save_path cnn_attack/`

### Attacker generates black-box adversarial testing sets
Assign the source model path as the Attacker's source CNN path. Generate adversarial testing sets.
`python adv_test.py --task MNIST_generate_black_box_data --fashion True --cnn_path cnn_attack/ --eps [0.1,0.15]`



## Training 
### Raw-train a Q-Model
`python rawtrain.py --task CIFAR_VQ --gpu 1 --save_best 1 --early_stop 40 --epochs 100 --update train --set_fixed 4 --num_concept 64 --inter_layer vgg16_pool0.5`

### Adv-train a Q-Model
`python adv_train.py --task CIFAR_VQ --gpu 1 --load_path ./save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1.0_1_10.0_10.0_lr_0.001_dim_16_share_True_update_train/ --last_name [WHICH_MODEL]  --eps 0.02 --early_stop 40 --save_best 1 --ratio 1 --epochs 100`

## Testing
### Black-box attacks (using the generated adversarial testing sets)
`python adv_test.py --task CIFAR --gpu 0 --vq_path ./save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1.0_1_10.0_10.0_lr_0.001_dim_16_share_True_update_train/ --FGSM_black True --BIM_black True`

### White-box attacks
`python adv_test.py --task CIFAR --gpu 0 --vq_path ./save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1.0_1_10.0_10.0_lr_0.001_dim_16_share_True_update_train/ --FGSM_white True --BIM_white True --CIM_white True --eps [0.02,0.05]`


