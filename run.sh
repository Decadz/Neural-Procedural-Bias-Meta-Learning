# Testing scripts for validating code correctness.
python experiments/run_maml.py --dataset omniglot --model conv4a --seeds 0 --fast True --base_gradient_steps 1

# Few-Shot Learning Experiments.
python experiments/run_maml.py --dataset omniglot --model conv4a --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset omniglot --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset miniimagenet --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_maml.py --dataset miniimagenet --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset fc100 --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_maml.py --dataset fc100 --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_maml.py --dataset tieredimagenet --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_maml.py --dataset tieredimagenet --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0


python experiments/run_transfer.py --dataset omniglot --model conv4a --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset omniglot --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset miniimagenet --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4 --device cuda:0
python experiments/run_transfer.py --dataset miniimagenet --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset fc100 --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_transfer.py --dataset fc100 --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0

python experiments/run_transfer.py --dataset tieredimagenet --model conv4b --num_shots 1 --meta_batch_size 4 --seeds 0 1 2 3 4  --device cuda:0
python experiments/run_transfer.py --dataset tieredimagenet --model resnet --num_shots 5 --meta_batch_size 2 --seeds 0 1 2 3 4 --device cuda:0
