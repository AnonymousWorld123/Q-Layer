from distribution import *
import pandas as pd

global config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.set_session(tf.Session(config=config))
names = ['CNN', 'VQ E-Path', 'VQ Q-Path']



# CIFAR
load_path = './save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1_1_0.001_0.001_lr_0.001_dim_16_share_True_update_train/'
last_name = '72_22536_q_0.7243_e_0.7070_loss_3.4791'
cnn_path = './save_val/cifar_01/cnn_baseline/'

# load_path = './save_val/cifar_01/fixed_vgg16_pool0.5_4_64_1_1_0.001_0.001_lr_0.001_dim_16_share_True_update_train/adv_e/'
# last_name = '84_26292_q_0.7866_e_0.8002_loss_1.4786'
# cnn_path = './save_val/cifar_01/cnn_baseline/adv_eps_0.02_ratio_1'

vq_hard = load_vq(load_path, config, last_name)
_, cnn = load_cnn(cifar_param.cnn, cnn_path, load_last=True)        

x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('cifar', norm = True)


attack_file_path = 'save_val/attacks/CIFAR_pool3_FGSM_copy_x_test_advs.npy'
cnn_attack_distance = cnn_distance_evaluate(cnn, attack_file_path, x_test, gaussian=False, dimensions = [0,64])
vq_attack_distance_e = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'e', gaussian=False, dimensions = [0,64])
vq_attack_distance_q = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'q', gaussian=False, dimensions = [0,64])

for key in cnn_attack_distance:
    print('---',key,'---')
    plot_distance_figure(cnn_attack_distance, vq_attack_distance_e, vq_attack_distance_q, key, name = 'cifar')
    for index, item in enumerate([cnn_attack_distance[key], vq_attack_distance_e[key], vq_attack_distance_q[key]]):
        print(names[index], item.mean())



# MNIST
load_path = 'save_val/mnist/hard/fixed_vgg16_pool0.5_1_64_1.0_1_0.001_0.001_lr_0.001_dim_64_share_True_update_train/'
last_name = None
cnn_path = 'save_val/mnist/cnn_baseline/'

vq_hard = load_vq(load_path, config, last_name)
_, cnn = load_cnn(mnist_param.cnn, cnn_path, load_last=True)        

x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = False, norm = True)


attack_file_path = 'save_val/attacks/MNIST_FGSM_copy_x_test_advs.npy'
x_k_test = get_zk(x_test, vq_hard, False).reshape(-1)
no_ground = x_k_test != np.argmax(np.bincount(x_k_test)[0])

cnn_attack_distance = cnn_distance_evaluate(cnn, attack_file_path, x_test, gaussian=False, dimensions = [0,64])
vq_attack_distance_e = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'e', gaussian=False, dimensions = [0,64])
vq_attack_distance_q = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'q', gaussian=False, dimensions = [0,64])

for key in cnn_attack_distance:
    print('---',key,'---')
    plot_distance_figure(cnn_attack_distance, vq_attack_distance_e, vq_attack_distance_q, key, name = 'fashion', no_ground = no_ground)
    for index, item in enumerate([cnn_attack_distance[key], vq_attack_distance_e[key], vq_attack_distance_q[key]]):
        print(names[index], item[no_ground].mean())      


# Fashion MNIST
load_path = './save_val/fashion/hard/fixed_vgg16_pool0.5_1_64_1_1_0.001_0.001_lr_0.001_dim_64_share_True_update_train/'
last_name = None
cnn_path = './save_val/fashion/cnn_baseline/'

vq_hard = load_vq(load_path, config, last_name)
_, cnn = load_cnn(mnist_param.cnn, cnn_path, load_last=True)        

x_train, y_train, x_test, y_test, x_val, y_val, _min, _max = load_train_eval_test('mnist', fashion = True, norm = True)


attack_file_path = 'save_val/attacks/Fashion_BIM_copy_x_test_advs.npy'

x_k_test = get_zk(x_test, vq_hard, False).reshape(-1)
no_ground = x_k_test != np.argmax(np.bincount(x_k_test)[0])

cnn_attack_distance = cnn_distance_evaluate(cnn, attack_file_path, x_test, gaussian=False, dimensions = [0,64], norm = False, mse = True)
vq_attack_distance_e = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'e', gaussian=False, dimensions = [0,64], norm = False, mse = True)
vq_attack_distance_q = vq_distance_evaluate(vq_hard, attack_file_path, x_test, path = 'q', gaussian=False, dimensions = [0,64], norm = False, mse = True)

for key in cnn_attack_distance:
    print('---',key,'---')
    plot_distance_figure(cnn_attack_distance, vq_attack_distance_e, vq_attack_distance_q, key, name = 'fashion', no_ground = no_ground)
    for index, item in enumerate([cnn_attack_distance[key], vq_attack_distance_e[key], vq_attack_distance_q[key]]):
        print(names[index], item.mean())


# Select one cluster to plot
from sklearn.decomposition import PCA
x_test_advs = np.load(attack_file_path, allow_pickle=True).item()

X = x_test
X_attacked = x_test_advs[0.1]
names = ['CNN', 'VQ E-Path', 'VQ Q-Path']


z_k = get_zk(X, vq_hard, False).reshape(-1,)
z_q = get_zq(X, vq_hard).reshape(-1, 64)
z_e = get_ze(X, vq_hard).reshape(-1, 64)
cnn_feature = get_cnn_feature(X, cnn).reshape(-1, 64)

z_k_attacked = get_zk(X_attacked, vq_hard, False).reshape(-1,)
z_q_attacked = get_zq(X_attacked, vq_hard).reshape(-1, 64)
z_e_attacked = get_ze(X_attacked, vq_hard).reshape(-1, 64)
cnn_feature_attacked = get_cnn_feature(X_attacked, cnn).reshape(-1, 64)

# lower dimensions
qmodel_pca = PCA(2).fit(z_e)
z_q_2 = qmodel_pca.transform(z_q)
z_e_2 = qmodel_pca.transform(z_e)

z_q_attacked_2 = qmodel_pca.transform(z_q_attacked)
z_e_attacked_2 = qmodel_pca.transform(z_e_attacked)

cnn_pca = PCA(2).fit(cnn_feature)
cnn_2 = cnn_pca.transform(cnn_feature)
cnn_2_attacked = cnn_pca.transform(cnn_feature_attacked)

selected_index = [10, 22]
plot_number = 10

mask_ = np.bitwise_and(z_k == z_k_attacked, z_k == selected_index[0])
mask__ = np.bitwise_and(z_k == z_k_attacked, z_k == selected_index[1])
selected_plot_ = random.sample(list(np.where(mask_)[0]), plot_number)
selected_plot__ = random.sample(list(np.where(mask__)[0]), plot_number)
selected_plot = selected_plot_ + selected_plot__

e = (z_e_2[selected_plot], z_e_attacked_2[selected_plot])
q = (z_q_2[selected_plot], z_q_attacked_2[selected_plot])
cnn = (cnn_2[selected_plot],cnn_2_attacked[selected_plot])

# plot
writer = pd.ExcelWriter('A.xlsx')
for index, item in enumerate([cnn, e, q]):
    f = plt.figure(figsize=(3,6))
    plt.scatter(item[0][:,0],item[0][:,1], label = 'Clean', c='b', alpha = 0.5)
    plt.scatter(item[1][:,0], item[1][:,1], label = 'Attacked', c='r', alpha = 0.5)
    data = pd.DataFrame(np.vstack((item[0], item[1])))
    data.to_excel(writer, names[index], float_format='%.5f')
    plt.title(names[index])
    plt.savefig('./distribution_' + names[index] + '.png',bbox_inches='tight',dpi=f.dpi)
    plt.close('all')

writer.save()

writer.close()




