import argparse

import wandb
import pprint
import numpy as np
import torch
import torch.optim as optim

from datasets.multimodal_features import create_multimodal_data, SequentialMultiModalFeatures
from main_sparse_train_w_data_gradient_efficient import sparCL
from utils.configuration_util import load_data_model


args = argparse.Namespace(
    arch='simple',
    arch_type = 'dynamic',
    patience=15,
    # arch='resnet',
    shuffle= False,
    # modal_file='HHAR/gyro_motion_hhar.pkl',
    modal_file='HHAR/accel_motion_hhar.pkl',
    buffer_mode='reservoir',
    depth=18,
    workers=4,
    multi_gpu=False,
    s=0.0001,
    batch_size=128,
    test_batch_size=256,
    epochs=500,
    optmzr='adam',
    lr=0.03,
    lr_decay=60,
    momentum=0.9,
    weight_decay=0.0001,
    no_cuda=False,
    seed=888,
    lr_scheduler='cosine',
    warmup=False,
    warmup_lr=0.0001,
    warmup_epochs=0,
    mixup=False,
    alpha=0.3,
    smooth=False,
    smooth_eps=0.0,
    log_interval=10,
    rho=0.0001,
    pretrain_epochs=0,
    pruning_epochs=0,
    remark='irr_0.75_mut',
    save_model='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500/',
    sparsity_type='random-pattern',
    config_file='config_vgg16',
    # use_cl_mask=True,
    use_cl_mask=False,
    buffer_size=800,
    buffer_weight=0.2, #TODO: original value was 0.1
    buffer_weight_beta=0.4, #TODO: original was 0.5
    # dataset='seq-cifar10',
    dataset='multi_modal_features',
    # dataset='hhar_features' if test_har else 'seq-cifar10',
    validation=False,
    test_epoch_interval=1,
    evaluate_mode=False,
    eval_checkpoint=None,
    gradient_efficient=False,
    gradient_efficient_mix=True,
    gradient_remove=0.1,
    gradient_sparse=0.8,
    sample_frequency=30,
    replay_method='derpp',
    patternNum=8,
    rand_seed=False,
    log_filename='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500//seed_888_75_derpp_0.80.txt',
    resume=None,
    save_mask_model=False,
    mask_sparsity=None,
    output_dir='checkpoints/resnet18/paper/gradient_effi/mutate_irr/seq-cifar10/buffer_500/',
    output_name='irr_0.75_mut_RM_3000_20',
    remove_data_epoch=20,
    data_augmentation=False,
    remove_n=3000,
    keep_lowest_n=0,
    sorting_file=None,
    input_dir='.',
    sp_retrain=True,
    sp_config_file='',
    sp_no_harden=False,
    sp_admm_sparsity_type='irregular',
    sp_load_frozen_weights=None,
    retrain_mask_pattern='weight',
    sp_update_init_method='zero',
    sp_mask_update_freq=5,
    sp_lmd=0.5,
    retrain_mask_sparsity=-1.0,
    retrain_mask_seed=None,
    sp_prune_before_retrain=True,
    output_compressed_format=False,
    sp_grad_update=False,
    sp_grad_decay=0.98,
    sp_grad_restore_threshold=-1,
    sp_global_magnitude=False,
    sp_pre_defined_mask_dir=None,
    upper_bound='0.74-0.75-0.75',
    lower_bound='0.75-0.76-0.75',
    mask_update_decay_epoch='5-45',
    cuda=True
)


wandb.login()

sweep_config = {
    'method': 'grid'
}
metric = {
    'name': 'f1_macro',
    'goal': 'maximize'
}
sweep_config['metric'] = metric




parameters_dict = {
    'buffer_weight':{
        'values': [0.005,0.01,0.03,0.05,0.1]
    },
    'buffer_weight_beta':{
        'values': [0.05,0.08,0.1,0.2, 0.03]
    }
    # 'features': {
    #     'values': [64, 128, 256, 512, 1024, 2048]
    # },
    # 'mask': {
    #     'values': [True, False]
    # },
    # # 'num_clusters':{
    # #     'values': [20,25,30,40,50]
    # # },
    # # 'batch_size': {
    # #     'values': [32,64,128,256,512,1024,2048]
    # # },
    # 'lr':{
    #     'values':[0.001, .0009, 0.03]
    # },
    # 'dropout':{
    #     'values': [0.1,0.2,0.3]
    # }

    # 'recon_size':{
    #     'values': [64, 128, 256, 512, 1024, 2048]
    # },
    # 'projection_size':{
    #     'values': [800, 1000, 1500, 2000, 3000]
    # }
}
# parameters_dict = {
#
# }

# parameters_dict = {
#     'learning_rate_scheduler':{
#         'values': ['none','linear', 'constant', 'exponential', 'step']
#     }
# }

sweep_config['parameters'] = parameters_dict
parameters_dict.update({
    'epochs': {
        'value': 1}
    })

combos = ((np.prod([len(parameters_dict[key]['values']) for key in parameters_dict if key != 'epochs'])))
# combos = ((np.prod([len(parameters_dict[key]['values']) for key in parameters_dict ])))


print("Tuning a total " + str(combos) + " combinations")

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="sweeping_downstream_spar")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configuration = {'type': 'multi_modal_clustering_features', 'files': '../SensorBasedTransformerTorch/extracted_features/multimodal_features_SHL.hkl', 'input_shape': 256, 'model': 'simple', 'epochs': 3, 'batch_size': 64, 'validation': False, 'sparsity_profile': 'profiles/in_out/irr/in_out_0.75.yaml'}
args.modal_file = '../SensorBasedTransformerTorch/extracted_features/multimodal_features_SHL.hkl'
data = create_multimodal_data(args)

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        lr = 0.03

        # data = load_datasets(['MotionSense', 'UCI', 'WISDM'], path='../datasets/processed/')
        features  = 128
        dropout = 0.2
        model, dataset = load_data_model(configuration, args)
        dataset = SequentialMultiModalFeatures(args,*data)
        buffer_weight = config.buffer_weight
        buffer_weight_beta = config.buffer_weight_beta
        args.use_cl_mask = False
        args.lr = lr
        args.buffer_weight = buffer_weight
        args.buffer_weight_beta = config.buffer_weight_beta
        optimizer = optim.Adam(model.parameters(), lr)


        model.to(device)
        _, _, all_scores = sparCL(args, model, dataset, 0)
        wandb.log({"f1_macro": all_scores["f1_macro"], "f1_micro": all_scores["f1_micro"]})

        # for epoch in range(config.epochs):
        #     avg_loss = train_epoch(network, data, optimizer, batch_size=config.batch_size, scheduler=scheduler, epoch=epoch)
        #     wandb.log({"loss": avg_loss, "epoch": epoch})


wandb.agent(sweep_id, train, count=combos)

