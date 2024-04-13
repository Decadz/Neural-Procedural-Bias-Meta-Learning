# Testing Scripts for validating code correctness.
python experiments/run_maml.py --dataset miniimagenet --model conv32 --num_ways 5 --num_shots 5 --meta_batch_size 2 --meta_gradient_steps 1000 --seeds 99 --fast True --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv32 --num_ways 5 --num_shots 5 --meta_batch_size 2 --meta_gradient_steps 1000 --seeds 99 --fast True --device cuda:0
python experiments/run_pretraining.py --dataset miniimagenet --model adaconv32 --pretraining_gradient_steps 1000 --seeds 0 --fast True --device cuda:0
python experiments/run_relation.py --dataset miniimagenet --model relationnet --meta_gradient_steps 1000 --seeds 0 --fast True --device cuda:0

# NPBML Few-Shot Learning Experiments.
python experiments/run_npbml.py --dataset miniimagenet --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset fc100 --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset cifarfs --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0  --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

python experiments/run_npbml.py --dataset tieredimagenet --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# Pretrained Backbone and Relation Network Experiments.
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

python experiments/run_relation.py --dataset miniimagenet --model relationnet --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset fc100 --model relationnet --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset cifarfs --model relationnet --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset tieredimagenet --model relationnet --seeds 0 --device cuda:0
