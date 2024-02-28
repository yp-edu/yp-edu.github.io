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
image: 
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

**Configuration attention:** In term of configuration attention the crux is to predict the first letter of the move. I'll mostly aim to interpret attention of this token. While I didn't conduct in depth analysis it seems than latter token are mostly doing pooling of the previous representation.

**Discovering simple heads:** The most simple heads will be located in the early layers. Interpreting latter heads is harder due to policementicity, convoluted circuits, high composition degree, you name it.

**L1 H7:** Focuses on the knights but only when it is paired in a token. This is where this tokenisation gets anoying.


![L1 H7](training-gpt2-on-stockfish-games_L1_H7.svg)
*Figure 1: L1 H7 configuration attention. FEN: `r1b1k1nr/pppp1ppp/2n1pq2/8/1b1PP3/2NQ4/PPP2PPP/R1B1KBNR w KQkq - 5 5`, Distribution: `Config`: 0.78 `Meta`: 0.10 `Dump`: 0.13.*
{: .im-center#v-relevance-flat}

## Resources

With this blog post I release everything needed to reproduce, explore, understand and extend this work. In summary:

- The model [->](https://huggingface.co/yp-edu/gpt2-stockfish-debug)
- The dataset [->](https://huggingface.co/datasets/yp-edu/stockfish-debug)
- The interactive space [->](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug)
- The code for the training and the space [->](https://github.com/yp-edu/training-gpt2-on-stockfish-games)

> [!quote] References
> 
> 1. Ruoss, Anian, et al. "Grandmaster-Level Chess Without Search." _ArXiv_, 2024, /abs/2402.04494.
