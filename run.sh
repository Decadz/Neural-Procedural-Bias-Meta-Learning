# Testing Scripts for validating code correctness.
python3 experiments/run_maml.py --dataset miniimagenet --model conv --num_ways 5 --num_shots 1 --meta_gradient_steps 1000 --seeds 0 --fast True 
python3 experiments/run_warpgrad.py --dataset miniimagenet --model linearwarpconv --num_ways 5 --num_shots 1 --meta_gradient_steps 1000 --seeds 0 --fast True 
python3 experiments/run_warpgrad.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_gradient_steps 1000 --seeds 0 --fast True
python3 experiments/run_testing.py --method maml --dataset miniimagenet --model conv --num_ways 5 --num_shots 1 --seeds 0 --fast True
python experiments/run_npbml.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --fast True --base_loss_fn learnedloss2

# MAML Few-Shot Learning Experiments.
python experiments/run_maml.py --dataset omniglot --model conv --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv --num_ways 5 --num_shots 5 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv --num_ways 20 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv --num_ways 20 --num_shots 5 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset miniimagenet --model conv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model conv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset fc100 --model conv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model conv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset cifarfs --model conv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model conv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset tieredimagenet --model conv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model conv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# T-Nets Few-Shot Learning Experiments.
python experiments/run_warpgrad.py --dataset omniglot --model linearwarpconv --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model linearwarpconv --num_ways 5 --num_shots 5 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model linearwarpconv --num_ways 20 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model linearwarpconv --num_ways 20 --num_shots 5 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset miniimagenet --model linearwarpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model linearwarpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model linearwarpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model linearwarpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset fc100 --model linearwarpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model linearwarpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model linearwarpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model linearwarpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset cifarfs --model linearwarpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model linearwarpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model linearwarpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model linearwarpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset tieredimagenet --model linearwarpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model linearwarpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model linearwarpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model linearwarpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# WarpGrad Few-Shot Learning Experiments (without warpgrad objective).
python experiments/run_warpgrad.py --dataset omniglot --model warpconv --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model warpconv --num_ways 5 --num_shots 5 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model warpconv --num_ways 20 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset omniglot --model warpconv --num_ways 20 --num_shots 5 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset miniimagenet --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset fc100 --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset fc100 --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset cifarfs --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_warpgrad.py --dataset cifarfs --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_warpgrad.py --dataset tieredimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_warpgrad.py --dataset tieredimagenet --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# NPBML Few-Shot Learning Experiments.
python experiments/run_npbml.py --dataset omniglot --model warpconv --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model warpconv --num_ways 5 --num_shots 5 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model warpconv --num_ways 20 --num_shots 1 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model warpconv --num_ways 20 --num_shots 5 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset fc100 --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset cifarfs --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset tieredimagenet --model warpconv --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model warpconv --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model warpresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model warpresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --base_bootstrapped_gradient_steps 5 --seeds 0 --device cuda:0

# Pretrained Backbone Experiments.
python experiments/run_pretraining.py --dataset omniglot --model conv --num_ways 5 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset omniglot --model conv --num_ways 20 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model conv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model conv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model conv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model conv --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset omniglot --model linearwarpconv --num_ways 5 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset omniglot --model linearwarpconv --num_ways 20 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model linearwarpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model linearwarpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model linearwarpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model linearwarpconv --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset omniglot --model warpconv --num_ways 5 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset omniglot --model warpconv --num_ways 20 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model warpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model warpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model warpconv --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model warpconv --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset miniimagenet --model resnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model resnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model resnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model resnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset miniimagenet --model linearwarpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model linearwarpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model linearwarpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model linearwarpresnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset miniimagenet --model warpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model warpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model warpresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model warpresnet --seeds 0 --device cuda:0
