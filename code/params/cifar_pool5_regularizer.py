import copy

mode = 'vgg16_pool5_regularizer'
save_path = './save_val/cifar_01/'

input_shape = (32,32,3)
pre_class = 10
lr = 0.001
heads = 4

dense_dim = 16
classifier_dim = 512
concept_dim = 16
regularizer = 0.0005
num_concept = 64

batch_size = 128

length = 40000

cnn = {
    'pre_class': pre_class,
    'classifier_dim':classifier_dim,
    'lr': lr,
    'input_shape':input_shape,
    'mode':mode,
    'batch_size':batch_size,
    'regularizer':regularizer,
    'save_path': save_path + 'cnn/',
    'optimizer':'Adam',
}

cnn_pool3 = {
    'pre_class': pre_class,
    'classifier_dim':classifier_dim,
    'lr': lr,
    'input_shape':input_shape,
    'mode':'vgg16_pool3_regularizer',
    'batch_size':batch_size,
    'regularizer':regularizer,
    'save_path': save_path + 'cnn_pool3/',
    'optimizer':'Adam',
}



hard = {
    'pre_class': pre_class,
    'dense_dim': dense_dim, 
    'concept_dim': concept_dim,   
    'classifier_dim':classifier_dim,
    'num_concept': num_concept,
    'c1':1,
    'c2':1,
    'alpha':0.001,
    'beta':0.001,
    'theta': 0,
    'gamma': 0,
    'lr': lr,
    'lr_decay_rate':0.3,
    'lr_decay_steps': 30 * length // batch_size,
    'set_att': 0,
    'set_trans': 0,
    'set_fixed':heads,
    'input_shape':input_shape,
    'mode':mode,
    'batch_size': batch_size,
    'distribute_size': None,
    'update':'train',
    'update_lr':0.001,
    'save_best':'q',
    'regularizer':regularizer,
    'early_stop':10,
    'save_path': save_path + 'hard/',
    'direct_pass':False,
    'optimizer':'AdamOptimizer',
    'omega':0,
}


hard_pool3 = copy.copy(hard)
hard_pool3['start_layer'] = 'vgg16_start'
hard_pool3['mode'] = 'vgg16_pool3'
hard_pool3['inter_layer'] = 'vgg16_pool0.5'
hard_pool3['omega'] = 0