python main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --lr_reduce 5 --epochs 100 --local_ep 2 --eval_every 1 --num_users 10 --frac 0.2 --iid --gpu 0 --timesteps 20 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test 



# SNN tests
python3 main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1  --momentum 0.95 --epochs 40 --local_ep 5 --num_users 5 --checkpoint_every 1 --frac 1 --iid --gpu 0 --timesteps 25 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1  --momentum 0.95 --epochs 40 --local_ep 5 --num_users 5 --checkpoint_every 1 --frac 1 --iid --gpu 1 --timesteps 25 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.01 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --snn --dataset CIFAR100 --num_classes 100 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1  --momentum 0.95 --epochs 40 --local_ep 5 --num_users 5 --checkpoint_every 1 --frac 1 --iid --gpu 4 --timesteps 25 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --snn --dataset CIFAR100 --num_classes 100 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1  --momentum 0.95 --epochs 40 --local_ep 5 --num_users 5 --checkpoint_every 1 --frac 1 --iid --gpu 5 --timesteps 25 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.01 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose

# ANN tests
python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 2 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --params_compress_rate 0.8 --result_dir test --verbose

python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 3 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --params_compress_rate 0.7 --result_dir test --verbose

python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 6 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --params_compress_rate 0.6 --result_dir test --verbose

python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 7 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --params_compress_rate 0.5 --result_dir test --verbose


# Not running 

python3 main_fed.py --dataset CIFAR100 --num_classes 100 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 6 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.02 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --dataset CIFAR100 --num_classes 100 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 5 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 7 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.03 --grad_rltv_noise_stdev 0.0 --result_dir test --verbose


# Debug
python3 main_fed.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001 0 --weight_decay 5e-4 --epochs 40 --local_ep 1 --eval_every 1 --checkpoint_every 1 --num_users 2 --frac 1 --iid --gpu 1 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed.py --snn --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 0 --epochs 40 --local_ep 1 --eval_every 1 --num_users 5 --frac 1 --iid --gpu 0 --timesteps 25 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir test --verbose

python3 main_fed-parallel.py --dataset CIFAR10 --num_classes 10 --model VGG9 --optimizer SGD --bs 32 --local_bs 32 --lr 0.001  --weight_decay 5e-4 --epochs 40 --local_ep 1 --checkpoint_every 1 --num_users 5 --frac 1 --iid --gpu 0,0,1,1,2,3 --grad_noise_stdev 0.0 --grad_abs_noise_stdev 0.0 --grad_rltv_noise_stdev 0.0 --params_compress_rate 1.0 --result_dir test --verbose
