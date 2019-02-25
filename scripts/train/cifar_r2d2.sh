expm_folder=cifar_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 20 --model.drop 0.4 --base_learner.init_adj_scale 1e1
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

