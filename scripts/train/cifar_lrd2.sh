expm_folder=cifar_lrd2_iter$1
echo $expm_folder
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 10 --data.test_way 5 --model.drop 0.4 --data.batch_size 200 --model.groupnorm False --train.learning_rate 5e-3 --train.lr_decay 0.5 --train.patience 200 --base_learner.method LRD2 --base_learner.iterations $1 --base_learner.init_lambda 50 --base_learner.linsys True --base_learner.init_adj_scale 10
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7
