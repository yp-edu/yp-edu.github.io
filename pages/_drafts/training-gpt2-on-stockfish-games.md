---
title: Training GPT-2 on Stockfish Games
tldr: I trained a GPT-2 model on Stockfish self-played games in the most naive way with no search, and it can decently play. The model is trained to output the next move given the FEN string of the board (single state). While I present some gotchas and caveats the results are decent for the amount of work and computing invested. I also present a basic attention visualiser parsing the attention of the text tokens to the board.
tags:
  - Chess
  - LLM
  - XAI
  - Attention
  - Training
references: 
aliases: 
crossposts: 
publishedOn: 
editedOn: 
authors:
  - "[[Yoann Poupart]]"
readingTime: 
image: /assets/images/training-gpt2-on-stockfish-games_thumbnail.webp
description:
---

![Training GPT-2 on Stockfish Games](training-gpt2-on-stockfish-games.webp)

> [!tldr] TL;DR
> 
> I trained a GPT-2 model on Stockfish self-played games in the most naive way with no search, and it can decently play. The model is trained to output the next move given the FEN string of the board (single state). While I present some gotchas and caveats the results are decent for the amount of work and computing invested. I also present a basic attention visualiser parsing the attention of the text tokens to the board.

> [!example] Table of content
> 
> - [Context](#context)
> 	- [DeepMind Paper](#deepmind-paper)
> 	-  [Personal Background](#personal-background)
> - [Training Method](#training-method)
> 	- [General Framing](#general-framing)
> 	- [Technical Details](#technical-details)
> 	- [Straight-Forward Improvements](#straight-forward-improvements)
> - [Inspecting the Model](#inspecting-the-model)
> 	- [Attention as an Interpretation](#attention-as-an-interpretation)
> 	- [Parsing Text Attention in the Board](#parsing-text-attention-in-the-board)
> 	- [Simple Interpretation](#simple-interpretation)
> - [Resources](#resources)

## Context

### DeepMind Paper

 Deepming recently released a paper where they achieved grand-master level without search [@1](#resources).

### Personal Background

Already trained LLMs. Already worked on chess models like lc0. Already explored XAI / MI.

## Training Method

### General Framing

Fine-tuning of a pre-trained model.

### Technical Details

The training code is available on the associated GitHub repository [training-gpt2-on-stockfish-games](https://github.com/yp-edu/training-gpt2-on-stockfish-games).

- 1 epoch on dataset
- 12h on one A100 (40GB)

>[!danger] Tokenisation Gotcha
>
>The tokenisation is not taylored to chess modelling.

### Straight-Forward Improvements


> [!tip] Conditioned Model
> 
> Recent framing of RL as a (self-)supervised learning problem showed that training an agent on sub-optimal trajectories can lead to optimal stragies should the agent be conditioned on a reward signal. For example this is the setup of Imitation Learning [@2](#resources) and Decision Transformer [@3](#resources).


## Inspecting the Model

### Attention as an Interpretation

A Gradio space is available on Hugging Face [->](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug).

>[!danger] Interpretation Gotcha
>
>While attention can be used to do simple interpretation the claim needs to be measured. Inded when interpreting the attention coefficients you say nothing about the values $O=A^TV$. This weighted sum needs to be interpreted as a whole for the conclusion to be rigorous.

### Parsing Text Attention in the Board  

In order to project attention on the board I simply parsed the attention per token and kept only the attention on pieces. The attention on the last part of the FEN string is summed into `Meta` while the rest is summed in `Dump`. This is a simple proxy to circumvent the unadapted tokenisation. 

>[!danger] Measure Gotcha
>
> Always think twice about what you're actually measuring. Here the heatmap only carries information about the piece configuration so my heuristic would be to leave out heatmaps where less than 50% of attention is put on pieces (`Config` metric).

Always look at the distribution and the raw attention figure. The raw attention can give you insight when not to trust grouped attention, i.e. when a token refers to more than one square.

### Simple Interpretation

>[!warning] Disclaimer
>
> The following section contain rough preliminary results, experiments and takes, thus everything should be taken with care.

**Configuration attention:** In term of configuration attention the crux is to predict the first letter of the move. I would mostly aim to interpret the attention of this token. While I didn't conduct in depth analysis it seems than latter token are mostly doing pooling of the previous representation.

**Discovering simple heads:** The most simple heads will be located in the early layers. Interpreting latter heads is harder due to policementicity, convoluted circuits, high composition degree, you name it.

**L1 H7:** Focuses on the knights but only when it is paired in a token, see the figure [1](#L1-H7). This is where this tokenisation gets annoying since it's hard to say what the model might have learned out of these paired tokens.


![L1 H7](training-gpt2-on-stockfish-games_L1_H7.svg)
*Figure 1: Configuration attention of the head 7 in layer 1. 
FEN: `r1b1k1nr/pppp1ppp/2n1pq2/8/1b1PP3/2NQ4/PPP2PPP/R1B1KBNR w KQkq - 5 5`, Distribution: `Config`: 0.78 `Meta`: 0.10 `Dump`: 0.13.*
{: .im-center#L1-H7}

**e2e3 opening:** This opening is by far my favourite (not a strong player here ^^), but out of the `262k` training games only `13k` opens with `e2e3`. The main response to this opening is `e7e5` with 4k games. I then like to push my queen to `f3` but since this combination is not in the training games it seems that the model blunders doing `c7c5`. Pushing the bishop to `c4` leads the model to respond moving the knight to `c6` letting the queen go for the checkmate (so in 4). This feels expected since most likely, as a predictor, the LLM learned in a Bayesian way 

$$p(a_4|s_4)=p(a_4|a_1, a_2, a_3)=p(a_4),$$

since there is no prior on this sequence. 

> [!tip] Counting Experiment
> 
> I think that counting all the games present in the dataset and comparing it the learned distribution could be a nice low hanging fruit experiment. The aim would be to examine the learned distribution on boards where the model has a prior against on boards where it has no prior. This toy experiment could be framed in the larger picture of prediction generalisation and coherence.

**c7c5 blunder:** I am not giving reason for the aforemonetionned blunder here but mostly pointing the limitation of doing interpretability with attention. Inspecting the attention of the last layer after `1. e3 e5 2. Qf3` most heads are strongly attenting to the pawn in `e5`, see figure [2](#L12-H2), but in the end the pawn `c7` is moved. By the way **L12 H2** seems to focus on "player to move best placed pawn".

![L1 H7](training-gpt2-on-stockfish-games_L12_H2.svg)
*Figure 2: Configuration attention of the head 2 in layer 12. 
FEN: `rnbqkbnr/pppp1ppp/8/4p3/8/4PQ2/PPPP1PPP/RNB1KBNR b KQkq - 1 2`, 
Distribution: `Config`: 0.95 `Meta`: 0.01 `Dump`: 0.04.*
{: .im-center#L12-H2}

**b8c6 second blunder:** First it seems to fail seeing any danger or threat and continue to focus on pieces that would develop his position (like the knight move). Then as a white, before the kill, the model doesn't find the checkmate with the queen (obviously out-of-distribution here). Important to note that all training games are long since the self-play means two player of the approximate same level.

> [!tip] Adaptative Strength Experiment
> 
> It could be interesting to see if such predictor model adapt its level to the adversary. Did it blunder because I played such weak moves (w.r.t. to the meta)? Does it play better is you use the book openings? This might be not straight forward since this correlates with out-of-distribution evaluation.

This analysis is only preliminary so feel free to continue it and send me feedback, the discussion happens in this Discord [thread](https://discord.com/channels/729741769192767510/1112497516928315442) on Eleuther AI.

## Resources

With this blog post I release everything needed to reproduce, explore, understand and extend this work. In summary:

- The model [->](https://huggingface.co/yp-edu/gpt2-stockfish-debug)
- The dataset [->](https://huggingface.co/datasets/yp-edu/stockfish-debug)
- The interactive space [->](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug)
- The code for the training and the space [->](https://github.com/yp-edu/training-gpt2-on-stockfish-games)

> [!quote] References
> 
> 1. Ruoss, Anian, et al. "Grandmaster-Level Chess Without Search." _ArXiv_, 2024, /abs/2402.04494.
> 2. Jonathan Ho et al. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.
> 3. Chen, Lili, et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." _ArXiv_, 2021, /abs/2106.01345.
