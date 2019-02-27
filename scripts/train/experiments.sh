expm_folder=mini_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 16 --model.drop 0.1 --base_learner.init_adj_scale 1e-4
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=mini_r2d2_64c
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 16 --model.drop 0.1 --base_learner.init_adj_scale 1e-4 --model.model_name RRNet_small
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 20 --model.drop 0.4 --base_learner.init_adj_scale 1e1
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_r2d2_64c
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 20 --model.drop 0.4 --base_learner.init_adj_scale 1e1 --model.model_name RRNet_small
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=omni5_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset omniglot --data.way 60 --data.test_way 5 --model.drop 0 --augm.rotation True --base_learner.init_adj_scale 10 --data.batch_size 600
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=omni20_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset omniglot --data.way 60 --data.test_way 20 --model.drop 0 --augm.rotation True --base_learner.init_adj_scale 10 --data.batch_size 600
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 20 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 20 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=omni5_r2d2_64c
python run_train.py --log.exp_dir $expm_folder --data.dataset omniglot --data.way 60 --data.test_way 5 --model.drop 0 --augm.rotation True --base_learner.init_adj_scale 10 --data.batch_size 600 --model.model_name RRNet_small
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=omni20_r2d2_64c
python run_train.py --log.exp_dir $expm_folder --data.dataset omniglot --data.way 60 --data.test_way 20 --model.drop 0 --augm.rotation True --base_learner.init_adj_scale 10 --data.batch_size 600 --model.model_name RRNet_small
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 20 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 20 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=mini_binary_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 2 --data.test_way 2 --model.drop 0.1 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_binary_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 2 --data.test_way 2 --model.drop 0.3 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=mini_binary_lrd2
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 2 --data.test_way 2 --model.drop 0.1 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500 --base_learner.method LRD2 --base_learner.iterations 10 --base_learner.init_lambda 1 --base_learner.linsys True --base_learner.init_adj_scale 1
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_binary_lrd2
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 2 --data.test_way 2 --model.drop 0.3 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500 --base_learner.method LRD2 --base_learner.iterations 10 --base_learner.init_lambda 1 --base_learner.linsys True --base_learner.init_adj_scale 1
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=mini_lrd2_iter1
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 10 --data.test_way 5 --model.drop 0.1 --data.batch_size 200 --model.groupnorm False --train.learning_rate 5e-3 --train.lr_decay 0.5 --base_learner.method LRD2 --base_learner.iterations 1 --base_learner.init_lambda 50 --base_learner.linsys True --base_learner.init_adj_scale 10
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_lrd2_iter1
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 10 --data.test_way 5 --model.drop 0.4 --data.batch_size 200 --model.groupnorm False --train.learning_rate 5e-3 --train.lr_decay 0.5 --train.patience 200 --base_learner.method LRD2 --base_learner.iterations 1 --base_learner.init_lambda 50 --base_learner.linsys True --base_learner.init_adj_scale 10
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=mini_lrd2_iter5
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 10 --data.test_way 5 --model.drop 0.1 --data.batch_size 200 --model.groupnorm False --train.learning_rate 5e-3 --train.lr_decay 0.5 --base_learner.method LRD2 --base_learner.iterations 5 --base_learner.init_lambda 50 --base_learner.linsys True --base_learner.init_adj_scale 10
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

expm_folder=cifar_lrd2_iter5
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 10 --data.test_way 5 --model.drop 0.4 --data.batch_size 200 --model.groupnorm False --train.learning_rate 5e-3 --train.lr_decay 0.5 --train.patience 200 --base_learner.method LRD2 --base_learner.iterations 5 --base_learner.init_lambda 50 --base_learner.linsys True --base_learner.init_adj_scale 10
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7
