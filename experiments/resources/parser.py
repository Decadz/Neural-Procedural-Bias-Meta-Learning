import ast


def register_configurations(parser):

    # Experiment settings.
    parser.add_argument("--description", required=False, type=str, nargs='+',
                        help="An description of the experiment which will be saved in the output results file.")
    parser.add_argument("--dataset", required=True, type=str,
                        help="The dataset to use in this experiment, see experiments/resources/datasets.")
    parser.add_argument("--model", required=True, type=str,
                        help="The model to use in this experiment, see source/models.")
    parser.add_argument("--seeds", required=True, type=int, nargs="+",
                        help="The reproducibility seed to use. Note, if multiple are provided, i.e. --seeds 1 2 3 4,"
                             "  multiple experiments will be performed.")
    parser.add_argument("--device", required=False, type=str,
                        help="The device to run the experiment on, default will be gpu:0 if available else cpu.")

    # Few-Shot Learning Settings.
    parser.add_argument("--num_ways", required=False, type=int,
                        help="The number of classes (i.e. ways) for each of the FSL problem in the inner loop.")
    parser.add_argument("--num_shots", required=False, type=int,
                        help="The number of support instances available for each class during training and testing.")
    parser.add_argument("--test_shots", required=False, type=int,
                        help="The number of out-of-sample instances used to validate testing performance.")

    # Meta Optimization used in Meta-Training.
    parser.add_argument("--meta_gradient_steps", required=False, type=int,
                        help="The number of meta gradients steps to take during meta training.")
    parser.add_argument("--meta_optimizer_name", required=False, type=str,
                        help="The meta optimizer used for learning in the outer loop.")
    parser.add_argument("--meta_batch_size", required=False, type=int,
                        help="The batch size used for learning in the outer loop during meta training.")
    parser.add_argument("--meta_scheduler_name", required=False, type=str,
                        help="The scheduler used on the meta learning rate in the outer loop.")

    # Base Optimization used in Meta-Training.
    parser.add_argument("--base_gradient_steps", required=False, type=int,
                        help="The number of base gradient steps to take in the inner loop.")
    parser.add_argument("--base_optimizer_name", required=False, type=str,
                        help="The base optimizer used for learning in the inner loop.")

    # Encoder/Model Pretraining.
    parser.add_argument("--pretrained_backbone", required=False, type=lambda x: (str(x).lower() == 'true'),
                        help="Whether to use a pretrained backbone encoder (found in source/models/pretrained).")
    parser.add_argument("--pretraining_gradient_steps", required=False, type=int,
                        help="The number of gradient steps taken when performing pretraining (does"
                             " not effect meta-learning related scripts).")
    parser.add_argument("--pretraining_batch_size", required=False, type=int,
                        help="The batch size used when performing pretraining (does not effect"
                             " meta-learning related scripts).")

    # Learning Objectives and Performance Metric.
    parser.add_argument("--performance_metric", required=False, type=str,
                        help="Performance metric used for reporting performance.")
    parser.add_argument("--meta_loss_fn", required=False, type=str,
                        help="The loss function used for evaluating performance in the outer loop.")
    parser.add_argument("--base_loss_fn", required=False, type=str,
                        help="The loss function used for evaluating performance in the inner loop.")

    # Experiment Settings.
    parser.add_argument("--fast", required=False, default=False, type=lambda x: (str(x).lower() == 'true'),
                        help="If true makes code run slightly faster but is no longer deterministic.")
    parser.add_argument("--input_channels", required=False, type=int,
                        help="The number of input channels used in the images.")
    parser.add_argument("--output_path", required=False, type=str,
                        help="The output path for the final results.")
    parser.add_argument("--verbose", required=False, type=int,
                        help="Setting for controlling the console output while running.")


def override_configurations(args, args_unknown, required_args, dataset_config, method_config):

    # Create a copy of the first dictionary to preserve its original values
    config = dataset_config.copy()

    # If the key is not in the result dictionary, add it with the corresponding value
    for key, value in method_config.items():
        if key not in config:
            config[key] = value

    # Iterating over all the known overridden arguments.
    for arg in vars(args):

        # Non empty arguments which have been manually provided.
        if getattr(args, arg) is not None and arg not in required_args:

            # Overriding the default hyper-parameter value.
            config[arg] = getattr(args, arg)

    # Iterating over all the unknown overridden arguments (those that are inside a dictionary).
    if len(args_unknown) != 0:

        # Iterating over the list in key value pairs.
        for key, arg in zip(args_unknown[0::2], args_unknown[1::2]):

            if "meta_optimizer" in key:
                key = key.replace("--meta_optimizer_", "")
                config["meta_optimizer_settings"][key] = ast.literal_eval(arg)

            elif "meta_scheduler" in key:
                key = key.replace("--meta_scheduler_", "")
                config["meta_scheduler_settings"][key] = ast.literal_eval(arg)

            elif "base_optimizer" in key:
                key = key.replace("--base_optimizer_", "")
                config["base_optimizer_settings"][key] = ast.literal_eval(arg)

            else:
                raise ValueError("Don't know how to parse", key, arg)

    return config

