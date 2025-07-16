---
title: MARL Cluster Training
tldr: Let's dive into MARL training with BenchMARL and how to scale up your experiments to a cluster. We'll cover environment and model customisation, as well as how to run your experiments on a cluster even without internet access.
tags:
  - MARL
  - Cluster
references: 
aliases: 
crossposts: 
publishedOn: 2025-07-17
editedOn: 2025-07-17
authors:
  - "[[Yoann Poupart]]"
readingTime: 10
image: /assets/images/marl-cluster-training_thumbnail.png
description: TL;DR> Let's dive into MARL training with BenchMARL and how to scale up your experiments to a cluster. We'll cover environment and model customisation, as well as how to run your experiments on a cluster even without internet access.
---

![MARL Cluster Training](marl-cluster-training.png)

> [!tldr] TL;DR
> 
> Let's dive into MARL training with BenchMARL and how to scale up your experiments to a cluster. We'll cover environment and model customisation, as well as how to run your experiments on a cluster even without internet access.

> [!example] Table of Contents
> 
> - [Context](#context)
> 	-  [BenchMARL](#benchmarl)
> - [MARL Training](#marl-training)
> 	- [Basic Setup](#basic-setup)
> 	- [Custom Task](#custom-task)
> 	- [Custom Model](#custom-model)
> - [Cluster Training](#cluster-training)
> 	- [Cluster Setup](#cluster-setup)
> 	- [Running Experiments](#running-experiments)
> 	- [No Internet](#no-internet)
> 	- [Results](#results)
> - [Resources](#resources)

## Context

As a part of [my PhD](https://yp-edu.github.io/stories/my-phd), I'm working on Multi-Agent Reinforcement Learning (MARL), more precisely on interpretability of MARL, as I outlined in [this article](https://arxiv.org/abs/2502.00726). Yet to be able to interpret any agent it's important to master MARL training. My primary goal is to spend as little time as possible training MARL agents. I also want to use classical MARL algorithms and environments to be able to contextualise my results with the state of the art, and after testing a few tools, I chose to go with [BenchMARL](https://github.com/facebookresearch/BenchMARL).

### BenchMARL

BenchMARL is a specialised library designed to ease MARL training. It provides a standardised interface that enables reproducibility and fair benchmarking across various MARL algorithms and environments.

> [!tip] Extra
>
> BenchMARL is really well packaged, easily extensible and already embeds configs' defaults.

BenchMARL's backends:

- TorchRL: provides a standardised interface for MARL algorithms and environments
- Hydra: provides a flexible and modular configuration system
- marl-eval: provides a standardised evaluation system

I encourage you to check out the [BenchMARL documentation](https://benchmarl.readthedocs.io/en/latest/) for more details.

## MARL Training

### Basic Setup

Along with this blog post, I have prepared a [repository](https://github.com/yp-edu/marl-cluster-training) with a basic setup of BenchMARL. Please refer to the README for installation instructions and details. You can also check some results on my [wandb report](https://wandb.ai/yp-edu/marl-cluster-training/reports/MARL-Cluster-Training--VmlldzoxMzQzMjU0Mg).

> [!success] Feedback
>
> Feel free to open an issue if you have any questions, problems or remarks you want to share.

The first thing you might want to do is train supported algorithms on a supported environment. Let's say you want to compare MAPPO and IPPO on Multiwalker. Such a benchmark is made of 2 independent experiments, that you can test individually using the script [`experiments/run_local.py`](https://github.com/yp-edu/marl-cluster-training/blob/main/scripts/experiments/run_local.py), inspired by the [`run.py`](https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/run.py) script provided by BenchMARL.

- IPPO & Multiwalker: \\
`uv run -m scripts.experiments.run_local algorithm=ippo task=pettingzoo/multiwalker`
- MAPPO & Multiwalker: \\
`uv run -m scripts.experiments.run_local algorithm=mappo task=pettingzoo/multiwalker`

These scripts are based on Hydra's configuration system, which allows you to easily modify the configuration of your experiment in a YAML file or through the command line. This is especially important when you want to run multiple experiments with different configurations, e.g. for hyperparameter search. Additionally, the defaults can be loaded directly from BenchMARL since the script's config (`exp:run_local.yaml`) adds it to the Hydra search path.

> [!tip] Tweaking
>
> You can easily tweak an algorithm by creating a new file in the `algorithm` config group and iterating on it manually or with a hyperparameter search. You can similarly tweak experiments and models.

But the strength of BenchMARL is to run benchmarks, i.e. a group of reproducible experiments with a similar config. You can start by running the [`benchmarks/multiwalker.py`](https://github.com/yp-edu/marl-cluster-training/blob/main/scripts/benchmarks/multiwalker.py) script, which is equivalent to running the previous experiments (with multiple seeds) but fully baked with the powerful plots from `marl-eval` to compare the experiments.

- Run the benchmark: `uv run -m scripts.benchmarks.multiwalker`

As this becomes tedious to run on a personal computer, the next step is to run it on a cluster. Jump to the [Cluster Training](#cluster-training) section for more details.

> [!success] Benchmark Config
>
> I proposed a [config](https://github.com/yp-edu/marl-cluster-training/blob/main/configs/bench:multiwalker.yaml) for the multiwalker benchmark based on object packing similar to how layers are composed. You can use it as a template to create your own benchmark configs.

### Custom Tas

One of BenchMARL's strengths is its ability to integrate custom tasks. First, let's say I want to create a supported task variation, e.g. Multiwalker with a shared reward.

> [!bug] Composition bug
>
> Unfortunately, unlike algorithms, tasks are nested in the framework folder (e.g. `pettingzoo`). Because of this and a bug in Hydra, it is not possible to compose defaults from nested tasks, see [this issue](https://github.com/facebookresearch/hydra/issues/3060). 

But it is still possible to derive a custom task in a few steps: 

- Create a config file: `multiwalker/shared.yaml`
- Register the task (before creating the experiment object): 

<script src="https://gist.github.com/Xmaster6y/0499dff7b7725f6160df03a58fccec6c.js"></script>

- Run the experiment: \\
`uv run -m scripts.experiments.run_local algorithm=mappo task=multiwalker/shared`

> [!danger] Hack
>
> Contrary to other groups (algorithms, experiments or models), tasks need to be registered as they are not spawned directly. Indeed, they are spawned through their factored environment class. 

Another kind of custom tasks are unsupported tasks from a supported environment class, for example, KAZ (Knights Archers Zombies) from PettingZoo. First, you need to create a custom task class:

<script src="https://gist.github.com/Xmaster6y/b9ea4c83c88789b45ddcee2ccdbc03d5.js"></script>

Which is used in the task:

<script src="https://gist.github.com/Xmaster6y/0916d6f7187191b251d8db7893cc3d6f.js"></script>

Then you can register the task:

<script src="https://gist.github.com/Xmaster6y/4ba879431ccd864dc96b298789946140.js"></script>

And additionally, you can validate the task using a `dataclass` to serve as a schema, which you would add in a `ConfigStore`:

<script src="https://gist.github.com/Xmaster6y/114c4b4ffc3c256ba7b830ffa766be14.js"></script>

For truly custom environments, see the [examples](https://github.com/facebookresearch/BenchMARL/tree/main/examples/extending/task). You might also want to dig into the [torchrl](https://pytorch.org/rl/stable/api/torchrl.envs.env.html) documentation first, to understand how to create your own environments.

> [!question] Question
> 
> How to handle the KAZ vector state `(B, N, F)`?
 
> [!answer] Answer
> 
> As `N` is not an agent dimension, you cannot directly use the `Mlp` (`(B, F)` inputs) model nor the `Cnn` (`(B, F)` or `(B, H, W, C)` inputs) model. You'll need either to modify the environment to output the correct shape or use a custom model based on a `Cnn`, `Mlp` or a flattening layer. 

### Custom Model

As noted in the previous section, KAZ vector state requires a custom model. The easiest will be to modify the `MlP` model, flattening any extra dimension. The basic idea is to introduce a new `num_extra_dims` parameter in the model config, which will be used to flatten the input tensor.

<script src="https://gist.github.com/Xmaster6y/c828476db539b3c72e02bd53b83552c2.js"></script>

This parameter will first be used in the `_perform_checks` method to check that the input tensor has the correct shape, and finally in the `_forward` method to simplify the input tensor:

<script src="https://gist.github.com/Xmaster6y/51e77d720c191f9c67866de020ba0f17.js"></script>

It then needs to come with a config file `extra_mlp.yaml` and should be registered in the `model_config_registry`:

<script src="https://gist.github.com/Xmaster6y/b6149b984003df83519d9e5517918e28.js"></script>

And that's it! You can now use your custom model in your experiments.

> [!success] Extra Dims
>
> See the refined PR [here](https://github.com/facebookresearch/BenchMARL/pull/211).

## Cluster Training

Now that we can run the experiments we want locally, let's scale up our experiments to a cluster.

> [!note] Tweaks and tricks 
>
> The more optimisations you'll want on the training process, the more you'll need to dig into the `torchrl` backend for more control over the environments and objects, and in the `torch` backend for more control over the models and tensors.

### Cluster Setup

After having tested a few different setups, I ended up settling on a full `uv` config. Here is my opinionated setup:

- [Optional] Setup `git` in your project
- [Optional] Link a GitHub repository
- Sync your project to the cluster (I prefer using a `git` remote for easy bidirectional edits but `rsync` or `scp` can be simpler)
- Use `uv sync` to install the dependencies on the cluster (this step can be delegated to the job, should you have an internet connection, see [No Internet](#no-internet))
- Run your jobs or notebooks on the cluster

You should do this on a Work or Draft partition, avoid installing in your home directory which might have a limited space (in terms of disk space and inodes). Always be aware of your cluster configuration and check with your admin in case of doubts.

> [!success] Pros
>
> - It's super easy to use once you're used to `uv`
> - The configuration is the same as the local one
> - Fully compatible with `slurm` jobs and `jupyter` notebooks/hubs
> - Works without internet on the nodes (see [No Internet](#no-internet))

> [!fail] Cons
>
> - You might miss package optimisations tailored for your cluster
> - It can consume a lot of inodes (you might need to remove old virtual environments)
> - Not super compatible with classical `cuda` echosystem

### Running Experiments

The easy part with the setup I presented is that the same script can be used to run your experiments locally or on a cluster. So, except for the `slurm` arguments, which might differ depending on your cluster, you can use the same script.

A typical `slurm` script, that you can launch using `sbatch launch/bench:multiwalker-jz.sh`, would look like this:

<script src="https://gist.github.com/Xmaster6y/fc467b0b45ea3727acb4a8613d57dfea.js"></script>

And to make use of the GPU you can just switch the experiment config by adding the argument `experiment=gpu`. It will simply load the default experiment config (`base_experiment`) and override the `gpu` config:

<script src="https://gist.github.com/Xmaster6y/7245814c6fb5337403cc2e6b4f466827.js"></script>

Now you're ready to launch a bunch of jobs doing wild hyperparameter search, with bigger models, bigger batch sizes, etc.

> [!info] JupyterHub
>
> You can find an example of how to run a notebook on a cluster JupyterHub in my [project template](https://github.com/yp-edu/research-project-template/tree/main/notebooks), this can vary depending on your cluster (example made for JeanZay).

### No Internet

Some clusters, for security reasons, don't allow internet access on the nodes (e.g. JeanZay in France). This can be a problem if you want to set up your environment (e.g. with `uv`) directly on the nodes (which can be easier). So let's see how we can easily transpose what we've seen so far to a cluster without an internet connection.

As noted in the [Cluster Config Setup](#cluster-config-setup) section, you should install your dependencies before starting your jobs. If you don't have an internet connection on the head node (never seen this though), you might try to transfer your local setup, or if allowed, use a Docker image.

Then, the only thing you need to do is to remove or disable the tools that require internet access. In our experiments, you just need to use `wandb` in "offline" mode (e.g. using `experiment=gpu_offline` in the script):

<script src="https://gist.github.com/Xmaster6y/52a8a593df6ae73b6d1bab48ca36c683.js"></script>

And when running a script with `uv`, you should use the `--no-sync` flag to avoid syncing your dependencies again. Depending on your use case, you might need to download your datasets to a special partition beforehand.

> [!bug] Wandb offline
>
> Unfortunately, wandb has a bug related to the config logging for offline runs, see [this issue](https://github.com/wandb/wandb/issues/6974). In order to circumvent this issue, you'll need to slightly modify the `logger` of the `Experiment` class in BenchMARL to pass the config to `wandb.init`, see [this PR](https://github.com/facebookresearch/BenchMARL/pull/216).

### Results

You'll find more detailed script launchers in the `launch` folder and in the commit history (I did quite a few tries). I was able to match results from the original paper [@1](#resources), and for some runs, get higher returns. I'll try to make a clean report on wandb when I find some time, which will be [here](https://wandb.ai/yp-edu/marl-cluster-training/reports/MARL-Cluster-Training--VmlldzoxMzQzMjU0Mg).

Some considerations if you want to go deeper. In this post, I only scratched the surface of the possibilities of BenchMARL. When using it to run pure experiments from YAML, you'll find some limitations about customisation, which are totally overcome by manipulating the Python classes directly. One of the most powerful ways to customise `Experiment` is through callbacks that can enable parameter scheduling (e.g. for `lr` or `batch_size`) or add custom pre/post-processing.

> [!tip] Further Customisation
>
> As you saw in the previous sections, BenchMARL is a really powerful tool to train MARL agents. However, there are still some things you might want to customise to fit your experiment's needs. And the beauty of BenchMARL is that you can keep all the features you want and simply add your secret sauce, whether it's an algorithm, an environment, or anything else.

## Resources

To learn more about BenchMARL and MARL in general, here are some valuable resources:

- [BenchMARL Documentation](https://benchmarl.readthedocs.io/en/latest/)
- [BenchMARL GitHub Repository](https://github.com/facebookresearch/BenchMARL)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [Hydra Configuration Framework](https://hydra.cc/)

BenchMARL's Discord community is also a great place to ask questions and share experiences with other users. Also, feel free to open an issue or a PR if you want to add or suggest an edit.

> [!quote] References
> 
> 1. Gupta, Jayesh K. et al. “Cooperative Multi-agent Control Using Deep Reinforcement Learning.” _AAMAS Workshops_ (2017).
