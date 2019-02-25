expm_folder=mini_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 16 --model.drop 0.1 --base_learner.init_adj_scale 1e-4
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7
