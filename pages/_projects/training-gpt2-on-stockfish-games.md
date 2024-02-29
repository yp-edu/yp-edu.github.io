---
title: Training GPT-2 on Stockfish Games
tldr: I trained a GPT-2 model on Stockfish self-played games in the most naive way, with no search, and it can play decently. The model is trained to output the next move given the FEN string of the board (single state). While I present some gotchas and caveats, the results are quite acceptable for the amount of work and computing invested. I also present a basic attention visualiser parsing the attention of the text tokens into the board.
tags:
  - Chess
  - LLM
  - XAI
  - Attention
  - Training
references: 
aliases: 
crossposts: 
publishedOn: 2024-02-29
editedOn: 2024-02-29
authors:
  - "[[Yoann Poupart]]"
readingTime: 15
image: /assets/images/training-gpt2-on-stockfish-games_thumbnail.webp
description: TL;DR> I trained a GPT-2 model on Stockfish self-played games in the most naive way, with no search, and it can play decently. The model is trained to output the next move given the FEN string of the board (single state). While I present some gotchas and caveats, the results are quite acceptable for the amount of work and computing invested. I also present a basic attention visualiser parsing the attention of the text tokens into the board.
---

![Training GPT-2 on Stockfish Games](training-gpt2-on-stockfish-games.webp)

> [!tldr] TL;DR
> 
> I trained a GPT-2 model on Stockfish self-played games in the most naive way, with no search, and it can play decently. The model is trained to output the next move given the FEN string of the board (single state). While I present some gotchas and caveats, the results are quite acceptable for the amount of work and computing invested. I also present a basic attention visualiser parsing the attention of the text tokens into the board.

> [!example] Table of content
> 
> - [Context](#context)
> 	- [DeepMind Paper](#deepmind-paper)
> 	-  [Personal Background](#personal-background)
> - [Training Method](#training-method)
> 	- [Technical Details](#technical-details)
> 	- [Straight-Forward Improvements](#straight-forward-improvements)
> - [Inspecting the Model](#inspecting-the-model)
> 	- [Attention as an Interpretation](#attention-as-an-interpretation)
> 	- [Parsing Text Attention in the Board](#parsing-text-attention-in-the-board)
> 	- [Simple Interpretation](#simple-interpretation)
> - [Resources](#resources)

## Context

### DeepMind Paper

 Deepming recently released a paper where they achieved grand-master level without search [@1](#resources). They used a decoder only trained on the board state ([FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) string) to predict whether the value of the board, the value of the action ($Q$ values) or the best action. I decided to take a similar approach but make it as simple as possible; I only trained the model to predict the next action (not necessarily the best).

With this approach, there is no need to compute tons of evals using the Stockfish engine, no need to filter out illegal moves and no need to use the fancy RL scheme (yes, I have been traumatised). There are some catches that come with this oversimplification that I present throughout the post, but I think it was a nice reframing.

### Personal Background

For the record, this project only took me 1 full day of work to set up and around 3 more days to release everything. The overhead is mostly making this blog post, but it's a good exercise for me, so it's definitely worth it.
I have already trained LLMs in the past, although I am far from being an expert. I have also worked on chess models, most recently [lc0](https://lczero.org/), and I am familiar with XAI / MI.

This project was not really a challenge for me; it was more of a hobby project I wanted to share.

## Training Method

### Technical Details

**General Framing:** Fine-tuning of a pre-trained model.

The training code is available on the associated GitHub repository [training-gpt2-on-stockfish-games](https://github.com/yp-edu/training-gpt2-on-stockfish-games) (mostly quick and dirty). It is really straightforward and self-explanatory, as I relied on the Hugging Face Trainer (all the complexity is under the hood). Maybe the contribution you'll find more worth is the Apptainer code to launch the training (not that difficult), which is cluster-friendly.

For the [dataset](https://huggingface.co/datasets/yp-edu/stockfish-debug) I used the simple format of prompt + completion:

```
{"prompt": "FEN: {fen}\nMOVE:", "completion": " {move}"}
```

**The run details:**

- 1 epoch of the dataset
- LR of $10^{-5}$
- 12h on one A100 (40GB)

>[!danger] Tokenisation Gotcha
>
>For simplicity, I didn't change the tokeniser, so the tokenisation, [BPE](https://huggingface.co/learn/nlp-course/en/chapter6/5), is not tailored for chess modelling. For example, `_r`  (`_` for whitespace) and `r` get encoded differently, and `nb` is encoded as a single token while representing a knight and a bishop. Also note that I used the raw FEN string, so `8` actually represents 8 empty squares; definitely problematic for the model to grasp a spatial representation of the board with positional encodings (it still manages somewhat).

As discussed in the next section, changing the tokenisation is a low-hanging fruit to improve the model.

### Straightforward Improvements

**Prompt Engineering:** Similarly to using your favourite chatbot, prompt engineering could be a simple way of making the model better. The model could be prompted (during training) with the strength,  [ELO](https://en.wikipedia.org/wiki/Elo_rating_system), of the player to move and the final result. On inference time, the model can then be prompted with the desired ELO and outcome. It obviously is limited; don't expect your model to reach ELO 10,000 with this method.

> [!tip] Conditioned Model
> 
> Recent framing of RL as a (self-)supervised learning problem showed that training an agent on sub-optimal trajectories can lead to optimal strategies should the agent be conditioned on a reward signal. For example, this is the setup of Imitation Learning [@2](#resources) and Decision Transformer [@3](#resources).

**Tokenisation:** I think one the best way to encode the board would be to only represent the board tokens, with 7 different tokens (`P`, `N`, `B`, `R`, `Q`, `K` and `X` for empty squares), then using vertical and horizontal positional encodings (like in ViT), as well as for meta information. A piece's colour encodings are different for each piece, and common encodings for castling rights are player colour, en passant, half-clock move, and full move number.

**Model:** A text encoder with a special pooling token, `[CLS]`, and an input sequence of length 64 representing the board using the aforementioned tokenisation. The output of the encoder would be fed to a classification head of size `1858`, minimal move encodings but still complete (DeepMind didn't get this right ^^ (neither in Alpha Zero for the record ^^)).

## Inspecting the Model

A Gradio space is available to interact with the model and its attention; check it out on Hugging Face [->](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug).

### Attention as an Interpretation

Attention is the most simple interpretation you can think of when studying Transformers. It is built into the model, and so it does reflect some internal knowledge of the model. In the autoregressive case (our case), you are often interested in "Why did my model predict this token?". Basically, looking at a row of the attention coefficients $A$, defined by equation $\ref{eq:attention}$, for the predicted query token and the past key tokens gives you an importance distribution on the past tokens.

$$
\begin{equation}
%\label{eq:attention}
A={\rm softmax}( QK^T) 
\end{equation}
$$

>[!danger] Interpretation Gotcha
>
>While attention can be used to do simple interpretation, the claim needs to be measured. Indeed, when interpreting the attention coefficients, you say nothing about the output values $O=AV$. This weighted sum needs to be interpreted as a whole for the conclusion to be more rigorous.

While [@4](#resources) claimed that attention is not an explanation on its own, [@5](#resources) conducted a more in-depth study pointing out the limitation of attention. You should especially remember the computation graph as its whole. Namely, attention heads will be merged and, in addition, composed via the residual stream, so the computation graph becomes a mess.

### Parsing Text Attention on the board  

In order to project attention on the board, I simply parsed the attention per token and kept only the attention on pieces. The attention on the last part of the FEN string is summed into `Meta` while the rest of the tokens are summed in `Dump`. This is a simple workaround and proxy to circumvent the unadapted tokenisation. 

>[!danger] Measure Gotcha
>
> Always think twice about what you're actually measuring. Here, the heatmaps only carry information about the piece configuration, so my heuristic would be to leave out heatmaps where less than 50% of attention is put on pieces (`Config` metric).

Always look at the distribution among the three categories of tokens and the raw attention figure. The raw attention can give you insight into when not to trust grouped attention, i.e. when a token refers to more than one square. This is especially important since it creates illusory patterns, so you should refrain from your deepest instinct to interpret this as a meaningful pattern (everyone loves to see patterns ^^).

### Simple Interpretation

>[!warning] Disclaimer
>
> The following section contains rough preliminary results, experiments and takes. Thus, everything should be taken with care.

**Configuration attention:** In terms of configuration attention, the crux is to predict the first letter of the move. I would mostly aim to interpret the attention of this token. While I didn't conduct an in-depth analysis, it seems that the latter tokens are mostly doing a pooling of the previous representation.

**Discovering simple heads:** The most simple heads will be located in the early layers. Interpreting later heads is harder due to polysemanticity, convoluted circuits, and a high degree of composition, you name it.

> [!success] Advice
> If you're not familiar with chess notation or attention, maybe the better is to interact with the associated Hugging Face [space](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug). By copying the [PGN](https://en.wikipedia.org/wiki/Portable_Game_Notation) (sequence of moves) or [FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) (state of the board) you can visualise the plot of this post and more. It's especially important if you disagree with my claims or don't trust me (you shouldn't ^^).

**L1 H7:** Focuses on the knights but only when it is paired in a token; see the figure [1](#L1-H7). This is where this tokenisation gets annoying since it's hard to say what the model might have learned from these paired tokens.

![L1 H7](training-gpt2-on-stockfish-games_L1_H7.svg)
*Figure 1: Configuration attention of the head 7 in layer 1.  
[FEN] `r1b1k1nr/pppp1ppp/2n1pq2/8/1b1PP3/2NQ4/PPP2PPP/R1B1KBNR w KQkq - 5 5`,  
[Distribution] `Config`: 0.78 `Meta`: 0.10 `Dump`: 0.13.*
{: .im-center#L1-H7}

**e2e3 opening:** This opening is by far my favourite (not a strong player here ^^), but out of the `262k` training games, only `13k` opens with `e2e3`. The main response to this opening is `e7e5` with 4k games. I then like to push my queen to `f3`, but since this combination is not in the training games, it seems that the model blunders doing `c7c5`. Pushing the bishop to `c4` leads the model to respond, moving the knight to `c6`, letting the queen go for the checkmate (so in 4). This feels expected since, most likely, as a predictor, the LLM learned in a Bayesian way 

$$p(a_4|s_4)=p(a_4|a_1, a_2, a_3)=p(a_4),$$

since there is no prior on this sequence. 

> [!tip] Counting Experiment
> 
> I think that counting all the games present in the dataset and comparing them to the learned distribution could be a nice low-hanging fruit experiment. The aim would be to examine the learned distribution on boards where the model has a prior against boards where it has no prior. This toy experiment could be framed in the larger picture of prediction generalisation and coherence.

**c7c5 blunder:** I am not giving the reason for the aforementioned blunder here but mostly pointing out the limitation of doing interpretability with attention. Inspecting the attention of the last layer after `1. e3 e5 2. Qf3` most heads are strongly attending to the pawn in `e5`, see figure [2](#L12-H2), but in the end, the pawn `c7` is moved. By the way, **L12 H2** seems to focus on "player to move best-placed pawn".

![L1 H7](training-gpt2-on-stockfish-games_L12_H2.svg)
*Figure 2: Configuration attention of the head 2 in layer 12.  
[FEN] `rnbqkbnr/pppp1ppp/8/4p3/8/4PQ2/PPPP1PPP/RNB1KBNR b KQkq - 1 2`,  
[Distribution] `Config`: 0.95 `Meta`: 0.01 `Dump`: 0.04.*
{: .im-center#L12-H2}

**b8c6 second blunder:** First, it seems to fail to see any danger or threat and continue to focus on pieces that would develop his position (like the knight move). Then, as a white, before the kill, the model doesn't find the checkmate with the queen (obviously out-of-distribution here). It is important to note that all training games are long since the self-play means two players of approximately the same level.

> [!tip] Adaptative Strength Experiment
> 
> It could be interesting to see if such a predictor model adapts its level to the adversary. Did it blunder because I played such weak moves (w.r.t. to the meta)? Does it play better if you use the book openings? This might not be straightforward since this correlates with out-of-distribution evaluation.

This analysis is only preliminary, so feel free to continue it and send me feedback; the discussion happens in this Discord [thread](https://discord.com/channels/729741769192767510/1112497516928315442) on Eleuther AI.

## Resources

With this blog post, I release everything needed to reproduce, explore, understand and extend this work. In summary:

- The model [->](https://huggingface.co/yp-edu/gpt2-stockfish-debug)
- The dataset [->](https://huggingface.co/datasets/yp-edu/stockfish-debug)
- The interactive space [->](https://huggingface.co/spaces/yp-edu/viz-gpt2-stockfish-debug)
- The code for the training and the space [->](https://github.com/yp-edu/training-gpt2-on-stockfish-games)

> [!quote] References
> 
> 1. Ruoss, Anian, et al. "Grandmaster-Level Chess Without Search." _ArXiv_, 2024, /abs/2402.04494.
> 2. Jonathan Ho et al. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.
> 3. Chen, Lili, et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." _ArXiv_, 2021, /abs/2106.01345.
> 4. Jain, Sarthak, and Byron C. Wallace. "Attention Is Not Explanation." _ArXiv_, 2019, /abs/1902.10186.
> 5. Wiegreffe, Sarah, and Yuval Pinter. "Attention Is Not Not Explanation." _ArXiv_, 2019, /abs/1908.04626.
