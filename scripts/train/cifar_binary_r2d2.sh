expm_folder=cifar_binary_r2d2
python run_train.py --log.exp_dir $expm_folder --data.dataset cifar100 --data.way 2 --data.test_way 2 --model.drop 0.3 --data.batch_size 32 --model.groupnorm True --train.learning_rate 0.001 --train.lr_decay 0.99 --train.patience 500
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 2 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

