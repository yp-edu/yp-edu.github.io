---
title: Layer-Wise Relevance Propagation
tldr: LRP is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
tags:
  - XAI
references: 
aliases: 
crossposts: 
publishedOn: 2024-01-16
editedOn: 2024-01-16
authors:
  - "[[Yoann Poupart]]"
readingTime: 8
image: /assets/images/layer-wise-relevance-propagation.png
description: TL;DR> LRP is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
---
> [!caution] WIP
> 
> This article is a work in progress.

![Layer-Wise Relevance Propagation](layer-wise-relevance-propagation.png)

> [!tldr] TL;DR
> 
> LRP is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.

> [!example] Table of content
> 
> - [LRP Framework](#lrp-framework)
> 	- [Formulation](#formulation)
> 	-  [Different Rules for Different Layers](#different-rules-for-different-layers)
> 	-  [Technical Details](#technical-details)
> - [Interpreting Othello Zero](#interpreting-othello-zero)
> 	- [Playing Othello](#playing-othello)
> 	- [Network Decomposition](#network-decomposition)
> 	- [Interpretation](#interpretation)
> 	- [Evaluate an Explanation](#evaluate-an-explanation)
> - [Resources](#resources)

## LRP Framework

### Formulation

LRP [@1](#resources) is a local interpretability method which attributes a relevance to any neuron activation inside a neural network with regard to a target output and for a given input. The target output can be non-terminal (e.g a intermediate layer neuron activation), multi-dimensional (e.g. the final vector of logits) or uni-dimensional (e.g. a single class logit). 

> [!info] Notation
> 
> The activation of the $j$-th neuron in the layer $l$ is noted $a_j^{[l]}$ and its associated relevance is noted $R_j^{[l]}$ and by convention $a_i^{[0]}$ represents the input. Weights are indexed similarly such that if the layer $l$ is linear its weights are noted $w_{ij}^{[l]}$ ([torch convention](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) and $b_i^{[l]}$ performing the mapping given by equation $\ref{eq:linear_mapping}$ where $a$ is the activation function.

$$
\begin{equation}
%\label{eq:linear_mapping}
a_i^{[l+1]}=a\left(\sum_{j}w_{ij}^{[l+1]}a_{j}^{[l]}+b_i^{[l+1]}\right)
\Leftrightarrow {\bf a}^{[l+1]}=a\left({\bf a}^{[l]}{ {\bf w}^{[l+1]}}^T+{\bf b}^{[l+1]}\right)
\end{equation}
$$

The figure [1](#relevance-backpropagation) illustrate the process of relevance propagation which can be intuited as a redistribution flow. The basic mechanism, LRP-$0$ [@1](#resources), is given by the equation $\ref{eq:propagation}$  which introduces the rule factor $z_{jk}^{[l]}$ of the layer $l$ and its associated normalisation factor $z_k^{[l]}=\sum_jz_{jk}^{[l]}$. For LRP-$0$ the rule is defined as $z_{jk}^{[l+1]}=a_j^{[l]}w_{kj}^{[l+1]}$ which can be interpreted in term of the classical Gradient$\times$Input [@2](#resources) for linear or ReLU activation. In practice the computation happens in a single modified backward pass.

$$
\begin{equation}
%\label{eq:propagation}
R_{j}^{[l]}=\sum_{k}\dfrac{z_{jk}^{[l+1]}}{z_k^{[l+1]}}R_k^{[l+1]}
\end{equation}
$$

> [!note] Rule
> 
> A rule  $z_{jk}^{[l]}$ defines how to redistribute the relevance $R_k^{[l]}$ of the layer $l$ into the earlier layers. The ideal rule depends on the nature of the layer $l$ and is an active topic of research [@3@4@5](#resources) (e.g. for "new" architecture of networks like Transformers), more on this in the [rule](#different-rules-for-different-layers) and [evaluation](](#evaluate-an-explanation) sections.

![layer-wise-relevance-propagation_backward](layer-wise-relevance-propagation_backward.png)
*Figure 1: Relevance back-propagation of the dog logit, [@7](#resources).*
{: .im-center#relevance-back-propagation}

The LRP-$0$ has nice properties like the flow conservation $\sum_jR_{j}^{[l]}=\sum_kR_{k}^{[l+1]}$ or the fact that it really tracks what was used by the model forward pass. Yet some gotchas were voluntarily dismissed for simplicity! First, as a classical backward pass, neuron's output relevance is propagated back into neuron's input relevance but also in the neuron's bias relevance. This can be circumvented by setting biases to zero (a trick). Then other rules like LRP-$\epsilon$ [@1](#resources) introduce $\epsilon$, a numerical stabilizer, which modify the normalisation factor as $z_k^{[l]}=\sum_jz_{jk}^{[l]}+\epsilon \cdot {\rm sign}\left(\sum_jz_{jk}^{[l]}\right)$. This trick is great but has the drawback to make the propagation mechanism not conservative.

>[!danger] Generalisation Gotcha
>
>The layer superscript is omitted in the literature because it doesn't cover the general formulation. For example to account for residual connection the propagation is spread across multiple layers. The general formulation is simply given by a causally weighted sum $R_{j}=\sum_{k}\omega_kR_k$ where the numbering is across all the neurons. It's definitely less informative about the actual computation, notation and interpretation!

### Different Rules for Different Layers

**[LRP-$\epsilon$ @1](#resources):** A

$$
\begin{equation}
%\label{eq:lrpe_rule}
R_{j}^{[l]}=\sum_{k}\dfrac{a_j^{[l]}w_{kj}^{[l+1]}}{\sum_ja_j^{[l]}w_{kj}^{[l+1]}+\epsilon \cdot {\rm sign}\left(\sum_ja_j^{[l]}w_{kj}^{[l+1]}\right)}R_k^{[l+1]}
\end{equation}
$$

**Flat :** j

$$
\begin{equation}
%\label{eq:flat_rule}
R_{j}^{[l]}=\sum_{k}\dfrac{1}{\sum_j 1}R_k^{[l+1]}
\end{equation}
$$

### Technical Details

- Backpass modification
- Backward hooks
- Stabilisers
- Cannonisers
- Input modifiers
- Weights modifiers

## Interpreting Othello Zero

### Playing Othello

Before digging into the actual interpretation of the network I borrowed from [Alpha Zero General](https://github.com/suragnair/alpha-zero-general) [@5](#resources), it is important to understand how it is used in practice and how it was trained. I highly recommend to check their code on [Github](https://github.com/suragnair/alpha-zero-general) or the associated [blog post](https://web.stanford.edu/~surag/posts/alphazero.html).

Tree representation of game (Min-Max, Alpha-Beta, MCTS, ...) is an intuitive representation of a game starting from the root (the current position), the nodes (board states $s$) and the edges (action chosen for a given state $(s,a)$). The Alpha Zero paper [@8](#resources) used MCTS PUCT, with the upper bound confidence (UCB) is given by the equation $\ref{eq:upper_confidence_boundary}$. This equation involves network predictions as $P(s,a)$ is the policy vector and $Q(s,a)$ is the average expected value over the visited children.

$$
\begin{equation}
%\label{eq:upper_confidence_boundary}
    U(s,a)=Q(s,a)+c_{\rm puct}\cdot P(s,a) \cdot \dfrac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
\end{equation}
$$

I reimplemented a simple functional version of this algorithm in the notebook, with few minor changes from the [Alpha Zero General](https://github.com/suragnair/alpha-zero-general) code.

<script src="https://gist.github.com/Xmaster6y/fd8ff108d39b0fdd09cb49e6809d2c54.js"></script>
### Network Decomposition

In order to use [Zennit](https://zennit.readthedocs.io/en/latest/) it is important to remember how it is implemented. Many difficulties arise in practice.

First all used modules should be instanciated under the target module (even activations). Then softmax should not be used because of the exponential. Here it can be safely removed as the output are the soflogmax which is a simple translation of the raw logits and doesn't change the softmax used after to select the next action.

It is important to acknowledge the similarity in computation of the value and the policy. This will lead to very close relevances heatmap as only one layer differ. 

One other practical limitation concern the empty cells. Using traditional LRP rules their relevance will be zero. Indeed during the computation the model doesn't use these pixels but it rather uses biases relevances. In order to overcome this difficulty it is possible to use a Flat rule, equation $\ref{eq:flat_rule}$, in the first layer to distribute equally the relevances among pixels.

### Interpretation

> [!warning] Disclaimer
> 
> The following experiments are highly shallow and I don't pretend they are highly relevant nor valuable. This work is only for illustrative purposes and definitely needs more digging. If you are interested for a follow-up of this project (by yourself or by me) and/or have questions, feel free to [contact](/about/#contact) me.

First here is the flat relevances induced from the first layer. This is fairly important as the flat rule will be used for the first layer in order to get relevances 

![bias_relevance](layer-wise-relevance-propagation_bias_relevance.png)
{: .im-center}

### Evaluate an Explanation

It is important to keep in mind that producing a heatmap is easy but interpreting it faithfully is hard.

It also was a critic of LRP with the DTD framing [@2](#resources) as interpreting an input-dependant heatmap is about interpreting an input and not really the model.

## Resources

A drafty notebook that self-contains all the practical experiments presented here and more is available on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ozMKtcRS9nRtvUfwZwj00ZZNpui5MhLr?usp=sharing). And bellow is a list of references containing the papers and code used in this post as well as additional resources.

- [@7](#resources) extends LRP to discover concepts.

> [!quote] References
> 
> 1. Bach, Sebastian, et al. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." _PLOS ONE_, vol. 10, no. 7, 2015.
> 2. Shrikumar, Avanti, et al. "Not Just a Black Box: Learning Important Features Through Propagating Activation Differences." _ArXiv_, 2016.
> 3. Binder, Alexander, et al. "Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers." _ArXiv_, 2016.
> 4. Lapuschkin, Sebastian, et al. "Unmasking Clever Hans Predictors and Assessing What Machines Really Learn." _Nature Communications_, vol. 10, no. 1, 2019.
> 5. Sixt, Leon, and Tim Landgraf. "A Rigorous Study Of The Deep Taylor Decomposition." _ArXiv_, 2022.
> 6. Thakoor, Shantanu, et al. "Learning to play othello without human knowledge." _Stanford University_, 2016.
> 7. Anders, Christopher J., et al. "Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy." _ArXiv_, 2021.
> 8. Achtibat, Reduan, et al. "From Attribution Maps to Human-understandable Explanations through Concept Relevance Propagation." _Nature Machine Intelligence_, vol. 5, no. 9, 2023.
> 9. Silver, David, et al. "Mastering the Game of Go Without Human Knowledge." Nature, vol. 550, no. 7676, 2017.
