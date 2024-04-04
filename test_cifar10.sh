python main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --lr_reduce 5 --epochs 100 --local_ep 2 --eval_every 1 --num_users 10 --frac 0.2 --iid --gpu 0 --timesteps 20 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test 



python3 main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr_interval '0.5 0.75' --lr 0.1 --lr_reduce 10 --epochs 40 --local_ep 5 --eval_every 1 --num_users 5 --frac 1 --iid --gpu 0 --timesteps 25 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test --verbose


python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr_interval '0.5 0.75' --lr 0.001 --lr_reduce 10 --weight_decay 5e-4 --epochs 40 --local_ep 5 --eval_every 1 --num_users 5 --frac 1 --iid --gpu 0 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test



# Debug
python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr_interval '0.5 0.75' --lr 0.001 --lr_reduce 10 --weight_decay 5e-4 --epochs 40 --local_ep 1 --eval_every 1 --checkpoint_every 1 --num_users 2 --frac 1 --iid --gpu 1 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr_interval '0.5 0.75' --lr 0.1 --lr_reduce 10 --epochs 40 --local_ep 1 --eval_every 1 --num_users 5 --frac 1 --iid --gpu 0 --timesteps 25 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test --verbose