# Q-Layer
Code for our paper "Q-Layer: Latent Space Contraints for Robust Convolutional Neural Network".

# To Use
The project is implemented with Tensorflow 1.15. Taking the Fashion-MNIST dataset as an example, we show how to train and test a Q-Model as follows:

## Prepare Dataset
Download the Fashion dataset to ./data/fashion

### Protector trains a validation CNN
`python rawtrain.py --task MNIST_CNN_1 --fashion True --early_stop 10 --epochs 40 --save_path cnn_validation/`

### Protector generates a validation set
Assign the source model path as the protector's validation CNN path. 

`python adv_test.py --task MNIST_generate_black_box_data --fashion True --cnn_path cnn_validation/ --eps [0.1] --validation_file True`

### Attacker trains a source CNN
`python rawtrain.py --task MNIST_CNN_1 --fashion True --early_stop 10 --epochs 40 --save_path cnn_attack/`

### Attacker generates black-box adversarial testing sets
Assign the source model path as the Attacker's source CNN path. Generate adversarial testing sets.

`python adv_test.py --task MNIST_generate_black_box_data --fashion True --cnn_path cnn_attack/ --eps [0.1,0.15]`



## Training 
### Raw-train a baseline CNN
`python rawtrain.py --task MNIST_CNN_2 --fashion True --early_stop 10 --epochs 40 --save_path cnn_baseline/`

### Raw-train a Q-Model
`python rawtrain.py --task MNIST_VQ --fashion True --early_stop 10 --epochs 40 --set_fixed 1 --num_concept 64 --inter_layer vgg16_pool0.5`

### Adv-train a Q-Model
Users can omit the "last_name" argument, which means loading the last checkpoint in the load path.

`python adv_train.py --task MNIST_VQ --fashion True --load_path ./save_val/fashion/hard/fixed_vgg16_pool0.5_1_64_1_1_0.001_0.001_lr_0.001_dim_64_share_True_update_train/ --last_name [CHECKPOINT NAME]  --eps 0.1 --early_stop 10 --epochs 40`

## Testing
### Black-box attacks (using the generated adversarial testing sets)
`python adv_test.py --task MNIST --fashion True --vq_path ./save_val/fashion/hard/fixed_vgg16_pool0.5_1_64_1_1_0.001_0.001_lr_0.001_dim_64_share_True_update_train/ --last_name [CHECKPOINT NAME] --FGSM_black True --BIM_black True`

### White-box attacks
`python adv_test.py --task MNIST --fashion True --vq_path ./save_val/fashion/hard/fixed_vgg16_pool0.5_1_64_1_1_0.001_0.001_lr_0.001_dim_64_share_True_update_train/ --last_name [CHECKPOINT NAME] --FGSM_white True --BIM_white True --CIM_white True --eps [0.1,0.15]`


