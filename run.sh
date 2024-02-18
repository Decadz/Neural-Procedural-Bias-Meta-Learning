# Testing Scripts for validating code correctness.
python experiments/run_maml.py --dataset miniimagenet --model conv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --meta_gradient_steps 1000 --seeds 99 --fast True --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --meta_gradient_steps 1000 --seeds 99 --fast True --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model adaconv48 --pretraining_gradient_steps 1000 --seeds 0 --fast True --device cuda:0

# MAML Few-Shot Learning Experiments.
python experiments/run_maml.py --dataset omniglot --model conv64 --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv64 --num_ways 5 --num_shots 5 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv64 --num_ways 20 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model conv64 --num_ways 20 --num_shots 5 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset miniimagenet --model conv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model conv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset fc100 --model conv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model conv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset cifarfs --model conv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model conv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_maml.py --dataset cifarfs --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_maml.py --dataset tieredimagenet --model conv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model conv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# NPBML Few-Shot Learning Experiments.
python experiments/run_npbml.py --dataset omniglot --model adaconv64 --num_ways 5 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model adaconv64 --num_ways 5 --num_shots 5 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model adaconv64 --num_ways 20 --num_shots 1 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset omniglot --model adaconv64 --num_ways 20 --num_shots 5 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset miniimagenet --model adaconv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset fc100 --model adaconv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaconv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset cifarfs --model adaconv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaconv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset tieredimagenet --model adaconv48 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaconv48 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# Pretrained Backbone Experiments.
python experiments/run_relation.py --dataset miniimagenet --model relationnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset omniglot --model adaconv64 --num_ways 5 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset omniglot --model adaconv64 --num_ways 20 --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset miniimagenet --model adaconv48 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model adaresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model wideadaresnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset fc100 --model adaconv48 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model adaresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset fc100 --model wideadaresnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset cifarfs --model adaconv48 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model adaresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset cifarfs --model wideadaresnet --seeds 0 --device cuda:0

python experiments/run_pretraining.py --dataset tieredimagenet --model adaconv48 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model adaresnet --seeds 0 --device cuda:0
python experiments/run_pretraining.py --dataset tieredimagenet --model wideadaresnet --seeds 0 --device cuda:0
