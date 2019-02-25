import configargparse
import distutils.util

from train import main

parser = configargparse.ArgumentParser(default_config_files=['conf/fewshots.yaml'],
                                       description='Train network for few-shot learning')

# # Data
parser.add('--data.dataset', type=str, metavar='{omniglot,imagenet,cifar100}', help="data set name")
parser.add('--data.root_dir', type=str, metavar='PATH', help="path to root data folder")
parser.add('--data.way', type=int, metavar='N', help="classes per episode, 0 for random")
parser.add('--data.shot', type=int, metavar='N', help="examples, 0 for random")
parser.add('--data.test_way', type=int, metavar='N', help="classes per episode in test")
parser.add('--data.train_episodes', type=int, metavar='N', help="train episodes per epoch")
parser.add('--data.test_episodes', type=int, metavar='N', help="test episodes per epoch")
parser.add('--data.batch_size', type=int, help="Batch size")
parser.add('--data.shot_min', type=int, help="When data.shot is 0, min number of (random) shots")
parser.add('--data.shot_max', type=int, help="When data.shot is 0, max number of (random) shots")
parser.add('--data.way_min', type=int, help="When data.way is 0, min number of (random) ways")
parser.add('--data.way_max', type=int, help="When data.way is 0, max number of (random) ways")
parser.add('--data.seed', type=int, help="Random seed")

# # Model
parser.add('--model.model_name', type=str,  metavar='{RRNet,RRNet_small}', help="embedding architecture")
parser.add('--model.lrelu', type=float,  metavar='FLOAT', help="Leaky ReLU slope")
parser.add('--model.drop', type=float,  metavar='FLOAT', help="Dropout prob")
parser.add('--model.groupnorm', type=distutils.util.strtobool, metavar='BOOL', help="BatchNorm or GroupNorm?")
parser.add('--model.bn_momentum', type=float,  metavar='FLOAT', help="BatchNorm forget factor")
parser.add('--model.debug', type=distutils.util.strtobool, metavar='BOOL')
parser.add('--model.model_path', type=str, metavar='PATH', help="path to pretrained model when testing")

# # Train
parser.add('--train.epochs', type=int, metavar='N', help='max epochs')
parser.add('--train.optim_method', type=str, help='optimization method')
parser.add('--train.learning_rate', type=float, metavar='FLOAT')
parser.add('--train.decay_every', type=int, metavar='N', help='epochs after which to decay the LR by lr_decay')
parser.add('--train.lr_decay', type=float, metavar='FLOAT', help='decay factor for learning rate')
parser.add('--train.weight_decay', type=float, metavar='FLOAT', help="weight decay")
parser.add('--train.patience', type=int, metavar='N', help='epochs to wait before validation improvement')
parser.add('--train.scheduler_type', type=str, metavar='{step,plateau}')
parser.add('--train.plateau_patience', type=int, metavar='N', help='epochs with no improvement to wait'
                                                                   'before decreasing the LR')

# # Logs
parser.add('--log.fields', type=str, metavar='field1,field2,...', help="fields to monitor during training")
parser.add('--log.exp_dir', type=str, metavar='DIR', help="directory where experiments"
                                                          "are saved (experiment date is automatically prepended)")

# # Run
parser.add('--data.cuda', type=distutils.util.strtobool, metavar='BOOL', help="run in CUDA mode")
parser.add('--data.gpu', type=int, metavar='N', help="GPU device (starts from 0)")

# # Data Augmentation
parser.add('--augm.rotation', type=distutils.util.strtobool, metavar='BOOL', help="augment with 4 rotations")
parser.add('--augm.crop', type=distutils.util.strtobool, metavar='BOOL', help="augment with random crops")
parser.add('--augm.max_crop_shrink', type=int, metavar='N', help="max shrinkage wrt original size during augm")
parser.add('--augm.cache_size', type=int, metavar='N', help="max number of augmented copies for each class")
parser.add('--augm.n_augment', type=int, metavar='N', help="multiplier of n_shots with augmented data")

# # Base-learner-specific params
parser.add('--base_learner.learn_lambda', type=distutils.util.strtobool, metavar='BOOL', help="Learn the"
                                                                                              "regularization coeff")
parser.add('--base_learner.init_lambda', type=float, metavar='FLOAT', help="regularizion coeff of Ridge Regression")
parser.add('--base_learner.init_adj_scale', type=float, metavar='FLOAT', help="output scaling init value")
parser.add('--base_learner.adj_base', type=float, metavar='FLOAT', help="base for adjust layer")
parser.add('--base_learner.lambda_base', type=float, metavar='FLOAT', help="base for lambda (when learnt)")
parser.add('--base_learner.linsys', type=distutils.util.strtobool, metavar='BOOL', help="If True uses torch.gesv"
                                                                                        "instead of inverting matrix")
parser.add('--base_learner.method', type=str, metavar='{R2D2,LRD2}')
parser.add('--base_learner.iterations', type=int, metavar='N', help="iterations for IRLS (only LR-D2)")


main(vars(parser.parse_args()))
