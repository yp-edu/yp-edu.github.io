---
title: Evidence of Learned Look-Ahead
tldr: Dive into how Leela Chess Zero, a class of neural networks to play chess, implements planning through a look-ahead mechanism. We'll break down the key findings of the linked article and explore the interpretability techniques used to study the model, like probing and activation patching. These findings will be reproduced using the library lczerolens, a tool for chess XAI.
tags:
  - XAI
  - AlphaZero
  - MI
references: 
aliases: 
crossposts: 
publishedOn: 2024-12-22
editedOn: 2024-12-22
authors:
  - "[[Yoann Poupart]]"
readingTime: 2
image: /assets/images/evidence-of-learned-look-ahead_thumbnail.png
description: TL;DR> Dive into how Leela Chess Zero, a class of neural networks to play chess, implements planning through a look-ahead mechanism. We'll break down the key findings of the linked article and explore the interpretability techniques used to study the model, like probing and activation patching. These findings will be reproduced using the library lczerolens, a tool for chess XAI.
---

![Evidence of Learned Look-Ahead](evidence-of-learned-look-ahead.png)

> [!tldr] TL;DR
>
> Dive into how Leela Chess Zero, a class of neural networks to play chess, implements planning through a look-ahead mechanism. We'll break down the key findings of the linked article and explore the interpretability techniques used to study the model, like probing and activation patching. These findings will be reproduced using the library `lczerolens`, a tool for chess XAI.

> [!example] Table of content
>
> - [Leela Chess Zero](#leela-chess-zero)
>   - [Chess-Playing Agents](#chess-playing-agents)
>   - [Model Architectures](#model-architectures)
> - [Cracking Open the Black Box](#cracking-open-the-black-box)
>   - [Attention Patterns](#attention-patterns)
>   - [Probing Analysis](#probing-analysis)
>   - [Activation Patching](#activation-patching)
> - [Exploration With `lczerolens`](#exploration-with-lczerolens)
>   - [Results Reproduction](#results-reproduction)
>   - [Analysing Other Models](#analysing-other-models)
>   - [What's Next?](#whats-next)
> - [Resources](#resources)


> [!info] Side Note
> 
> This post will explore the findings in [@1](#resources) ("Evidence of Learned Look-Ahead in a Chess-Playing Neural Network") and port them to my chess XAI library [`lczerolens`](https://github.com/Xmaster6y/lczerolens). The main goal will be to make it easy to extend the experiments of the article with different XAI methods or on different models, nonetheless I encourage any interested reader to have a close look at the article and their code available [here](https://leela-interp.github.io/).

## Leela Chess Zero

### Chess-Playing Agents

Leela Chess Zero (lc0) is a neural network-powered chess engine trained via reinforcement learning and self-play [@2][#resources]. Originally it was an open source reproduction of the groundebreaking algorithm AlphaZero [@3](#resources), but since the launch of this project the Leela team pushed many improvements like the latest transformer series of models [@4](@resources). At the heart of the lc0 chess engine there are two main components, the neural network, which can be modelled has an advanced search heuristic, and the Monte Carlo Tree Search (MCTS), which is a brute force operator that simulates game trajectories in order to find the best strategy.

The experiments from the article are only meant to study the network part, leaving the MCTS aside. Indeed, trying to interpret the MCTS along with the network can be harder, but understanding the standalone complex heuristic is already an interesting challenge. This network takes in input the board squares as tokens, process them in a Transformer architecture and then produces three outputs through the policy head, the value head and the move-left head. In practice these outputs then serve as a complex heuristic that guides the MCTS by a smart initialisation on new nodes.

> [!caution] Disclaimer
> 
> I won't enter much more in the details of the chess engine nor the training process. I'll refer the interested reader to look at the additional linked [resources](#resources).

### Model Architectures

One of the underrated artifacts produced by the Leela team is the astonishing number of different networks. They trained models with various architectures, number of layers, hidden sizes, number of attention heads; and they provide a ton of training checkpoints. This makes it one of the best interpretability testbed project, a true drosophila for XAI researcher. 

> [!tip] Future Track
> 
> A possible future track, IMO very interesting, would be to do a thorough benchmark study across the various trained models. Comparing the features learned with respect to the training time and the model size might yield valuable insights, singular learning theory could prove very handy here.

**CNN-Based** models were originally chosen for chess for AlphaZero and lc0 as they were proven efficient to deal with the structured data of games resembling images (Transformer came after, in 2017, and ViT only in 2020). In fact, CNN are still very common in modern deep RL for environment that can make use of vision like games or robotics, with existing foundation models, as they are efficient and often enough for simple environments.

**Transformer-Based** architecture has long been avoided in chess compared to other domains like NLP. One of the main reason was the difficulty to learn the key board features while being more computationally heavy [@5](#resources). As shown in [@5](#resources) and also in the new architectures proposed in [@4](#resources), feature/architecture engineering tailored to learning with AlphaZero is still an important challenge. Some of the ameliorations to the ViT architecture proposed by the Leela team are described in a [blogpost](https://lczero.org/blog/2024/02/transformer-progress/).

> [!success] Getting the Model
> 
> The experiments in [@1](#resources) are conducted on a Transformer-based architecture, precisely the one tagged as `T82-768x15x24h-swa-5230000` (weights `768x15x24h-t82-swa-5230000.pb.gz`). As the best nets [page](https://lczero.org/dev/wiki/best-nets-for-lc0/) is regularly updated one can look in their [public storage](https://storage.lczero.org/files/) and use a related network, e.g., `768x15x24h-t82-swa-5264000.pb.gz`.

## Cracking Open the Black Box

### Attention Patterns

In a Transformer model it is often possible to draw some explanations by simply looking at the attention patterns without much intervention. While some patterns can be hard to construe (for a human), it's a good metric for information movement. The "QK" circuit select the information to be moved while the "OV" circuit compute the information, see [@9](#resources) for how to think about attention heads in term of circuits. 

$$
\begin{equation}
%\label{eq:attention}
{\rm Attn}_i(x)={\rm softmax}(x^TW_{Qi}^TW_{Ki}x)\cdot W_{Oi}W_{Vi}x
\end{equation}
$$

Attention heads in the lc0 network reveal how the model connects different board squares. It is thus natural to see a lot of attention head exhibiting patterns related to specific piece moves (pawn/bishop/knight...). For example, if a square computes something about a bishop (attack, protection, sacrifice, etc.), this computation might be relevant to other squares reachable moving bishop-like.

> [!danger] Limitations
> 
> Interpreting attention patterns can be limited in term of conclusion. You should keep in mind that when you interpret the attention weights (from the softmax/"QK" circuit), you only look at the selection process and can't say anything by the actual selected computation.

### Probing Analysis

Probing involves training smaller models to extract information from LCZero's activations. For instance, a simple probe can predict optimal moves two turns into the future with over 90% accuracy, showing that LCZero encodes future plays within its layers.

In order to improve the accuracy it could be tempting to augment the probe complexity, to allegedly capture the underlying representation. Yet this might lead the probe to learn other correlations, coming from the data rather than the inner model's representation. For this reason it is always a good idea to have control tasks [@7](#resources), or a control model, e.g., randomly initialised like in [@7](#resources), known as the model parameter randomisation test [@8](#resources).

### Activation Patching

Activation patching was proposed a causal mediation analysis in order to causally measure the impact of different model's components to downstream tasks  [@6](#resources).

![activation-patching](evidence-of-learned-look-ahead_activation-patching.png)
*Figure 1: Illustration of the activation patching method. This method needs two forward passes, one clean (a) and one corrupted (b). Then to measure the impact of a component $C$ on a downstream task we can use an indirect intervention (c) or a direct intervention (d).*
{: .im-center#activation-patching}


It lets us test which parts of LCZero's neural activations are critical. By swapping activations between clean and corrupted board states, researchers pinpointed layers and squares that store crucial information about future moves.

## Exploration With `lczerolens`

### Results Reproduction

If you want to understand the dataset generation process read appendices D & E of the original article [@1](#resources). They used selection criteria based on weaker instances of the studied model and automatically generated corrupted states, pretty canny!

Using the `lczerolens` library, you can replicate key experiments like activation patching and probing. The library provides tools for extracting activations, visualizing attention, and conducting causal tests.

**Attention Patterns**

**Probing Analysis**

**Activation Patching**

### Analysing Other Models

_Empty for now_

### What's Next?

_Empty for now_

## Resources

In order to understand better the AlphaZero algorithm:

- Stanford blog post
- One of my previous blog post
- The original article of AlphaZero
- The video by DeepMind

In order to understand better how the MCTS is used in Leela:

- The original article of AlphaZero
- The blog of Leela

Other related interpretability paper on chess:
- Two papers by the interpretability team of deepmind
- A 
- My preliminary work using sparse autoencoders

> [!quote] References
>
> 1. Jenner, Erik, et al. "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network." 2024.
> 2. Pascutto, Gian-Carlo et al. Leela chess zero, 2019. URL http://lczero.org/.
> 3. Silver, David et al. “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play.” _Science_ 362 (2018): 1140 - 1144.
> 4. Monroe, Daniel and The Leela Chess Zero Team. “Mastering Chess with a Transformer Model.” _ArXiv_ abs/2409.12272 (2024).
> 5. Czech, Johannes et al. “Representation Matters for Mastering Chess: Improved Feature Representation in AlphaZero Outperforms Switching to Transformers.” _European Conference on Artificial Intelligence_ (2023).
> 6. Vig, Jesse et al. “Investigating Gender Bias in Language Models Using Causal Mediation Analysis.” _Neural Information Processing Systems_ (2020).
> 7. Hewitt, John and Percy Liang. “Designing and Interpreting Probes with Control Tasks.” _ArXiv_ abs/1909.03368 (2019): n. pag.
> 8. Adebayo, Julius et al. “Sanity Checks for Saliency Maps.” _Neural Information Processing Systems_ (2018).
> 9. Elhage, et al., "A Mathematical Framework for Transformer Circuits", Transformer Circuits Thread, 2021.