expm_folder=mini_binary_lrd2
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 2 --data.test_way 2 --model.drop 0.1 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500 --base_learner.method LRD2 --base_learner.iterations 10 --base_learner.init_lambda 1 --base_learner.linsys True --base_learner.init_adj_scale 1
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

