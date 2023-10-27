import ast


def register_configurations(parser):

    # Few-Shot Learning Settings.
    parser.add_argument("--num_ways", required=False, type=int)
    parser.add_argument("--num_shots", required=False, type=int)
    parser.add_argument("--test_shots", required=False, type=int)

    # Meta Optimization used in Meta-Training.
    parser.add_argument("--meta_gradient_steps", required=False, type=int)
    parser.add_argument("--meta_optimizer_name", required=False, type=str)
    parser.add_argument("--meta_batch_size", required=False, type=int)
    parser.add_argument("--meta_scheduler_name", required=False, type=str)

    # Base Optimization used in Meta-Training.
    parser.add_argument("--base_gradient_steps", required=False, type=int)
    parser.add_argument("--base_optimizer_name", required=False, type=str)
    parser.add_argument("--base_bootstrapped_gradient_steps", required=False, type=int)
    parser.add_argument("--base_bootstrapped_optimizer_name", required=False, type=str)

    parser.add_argument("--pretraining_gradient_steps", required=False, type=int)
    parser.add_argument("--pretraining_optimizer_name", required=False, type=str)
    parser.add_argument("--pretraining_batch_size", required=False, type=int)
    parser.add_argument("--pretraining_scheduler_name", required=False, type=str)

    # Inner and Outer Objectives.
    parser.add_argument("--performance_metric", required=False, type=str)
    parser.add_argument("--task_loss_function", required=False, type=str)
    parser.add_argument("--matching_loss_function", required=False, type=str)

    # Experiment Settings.
    parser.add_argument("--fast", required=False, default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--track_running_stats", required=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--input_channels", required=False, type=int)
    parser.add_argument("--output_path", required=False, type=str)
    parser.add_argument("--verbose", required=False, type=int)


def override_configurations(args, args_unknown, required_args, config):

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

            elif "base_bootstrapped_optimizer" in key:
                key = key.replace("--base_optimizer_", "")
                config["base_optimizer_settings"][key] = ast.literal_eval(arg)

            elif "pretraining_optimizer" in key:
                key = key.replace("--init_optimizer_", "")
                config["init_optimizer_settings"][key] = ast.literal_eval(arg)

            elif "pretraining_scheduler" in key:
                key = key.replace("--init_scheduler_", "")
                config["init_scheduler_settings"][key] = ast.literal_eval(arg)

            else:
                raise ValueError("Don't know how to parse", key, arg)
