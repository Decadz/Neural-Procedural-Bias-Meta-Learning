# Testing scripts for validating code correctness.
python experiments/run_maml.py --dataset miniimagenet --model conv4 --seeds 0 --fast True --base_gradient_steps 1 --meta_gradient_steps 1000
python experiments/run_npbml.py --dataset miniimagenet --model conv4 --seeds 0 --fast True --base_gradient_steps 1 --meta_gradient_steps 1000
python experiments/run_maml.py --dataset miniimagenet --model conv4 --seeds 0 --fast True --base_gradient_steps 1 --base_bootstrapped_gradient_steps 5 --meta_gradient_steps 1000

# NPBML Few-Shot Learning Experiments.
python experiments/run_npbml.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 5 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_npbml.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model warpconv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model warpconv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_npbml.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model warpconv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model warpconv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_npbml.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model warpconv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model warpconv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0


# MAML Few-Shot Learning Experiments.
python experiments/run_maml.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 5 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 1 --base_bootstrapped_gradient_steps 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0


# MAML with pre-training Few-Shot Learning Experiments.
python experiments/run_transfer.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 1 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset omniglot --model conv4 --num_ways 5 --num_shots 5 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset miniimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_transfer.py --dataset fc100 --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset fc100 --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_transfer.py --dataset fc100 --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset tieredimagenet --model conv4 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
