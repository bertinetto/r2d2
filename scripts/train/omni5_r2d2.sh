expm_folder=omni5_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset omniglot --data.way 60 --data.test_way 5 --model.drop 0 --augm.rotation True --base_learner.init_adj_scale 10 --data.batch_size 600
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

