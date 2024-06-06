# Download the required datasets (not including tiered-ImageNet)
python experiments/resources/datasets/download_datasets.py

# MiniImageNet 4-CONV 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset miniimagenet --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset miniimagenet --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# MiniImageNet ResNet-12 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset miniimagenet --model adaresnet --pretraining_optimizer_weight_decay 0.01 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset miniimagenet --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset miniimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# TieredImageNet 4-CONV 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset tieredimagenet --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset tieredimagenet --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# TieredImageNet ResNet-12 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset tieredimagenet --model adaresnet --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset tieredimagenet --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset tieredimagenet --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# CIFAR-FS 4-CONV 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset cifarfs --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset cifarfs --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# CIFAR-FS ResNet-12 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset cifarfs --model adaresnet --pretraining_optimizer_weight_decay 0.01 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset cifarfs --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset cifarfs --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# FC-100 4-CONV 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset fc100 --model adaconv128 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset fc100 --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaconv128 --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaconv128 --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0

# FC-100 ResNet-12 5-way 5-shot and 5-way 1-shot experiments.
python experiments/run_pretraining.py --dataset fc100 --model adaresnet --pretraining_optimizer_weight_decay 0.01 --seeds 0 --device cuda:0
python experiments/run_relation.py --dataset fc100 --model relationnet --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 1 --meta_batch_size 4 --seeds 0 --device cuda:0
python experiments/run_npbml.py --dataset fc100 --model adaresnet --num_ways 5 --num_shots 5 --meta_batch_size 2 --seeds 0 --device cuda:0
