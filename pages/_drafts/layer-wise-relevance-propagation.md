---
title: Layer-Wise Relevance Propagation
tldr: Layer-Wise Relevance Propagation (LRP) is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
tags:
  - XAI
  - AlphaZero
references: 
aliases: 
crossposts: 
publishedOn: 2024-01-16
editedOn: 2024-01-16
authors:
  - "[[Yoann Poupart]]"
readingTime: 12
image: /assets/images/layer-wise-relevance-propagation.png
description: TL;DR> Layer-Wise Relevance Propagation (LRP) is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
---
> [!caution] WIP
> 
> This article is a work in progress.

![Layer-Wise Relevance Propagation](layer-wise-relevance-propagation.png)

> [!tldr] TL;DR
> 
> Layer-Wise Relevance Propagation (LRP) is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.

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

LRP [@1](#resources) is a local interpretability method which attributes a relevance to any neuron activation inside a neural network with regard to a target output and for a given input. The target output can be non-terminal (e.g an intermediate layer neuron activation), multi-dimensional (e.g. the final vector of logits) or uni-dimensional (e.g. a single class logit). It's formulation is similar to the classical Gradient$\times$Input [@2](#resources) as it is computed using a single modified backward pass. Yet it doesn't suffer from the noisiness of the gradient.

> [!info] Notation
> 
> The activation of the $j$-th neuron in the layer $l$ is noted $a_j^{[l]}$ and its associated relevance is noted $R_j^{[l]}$ and by convention $a_j^{[0]}$ represents the input. Weights are indexed similarly such that if the layer $l+1$ is linear its weights are noted $w_{kj}^{[l+1]}$ ([torch convention](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)) and $b_k^{[l+1]}$ performing the mapping, to the $k$-th neuron of the layer $l+1$, given by equation $\ref{eq:linear_mapping}$ where $g$ is the activation function.

$$
\begin{equation}
%\label{eq:linear_mapping}
a_k^{[l+1]}=g\left(\sum_{j}w_{kj}^{[l+1]}a_{j}^{[l]}+b_k^{[l+1]}\right)
\end{equation}
$$

The figure [1](#relevance-back-propagation) illustrate the process of relevance propagation which can be intuited as a redistribution flow. The relevance $R_k^{[l+1]}$ is decomposed in "messages" sent to the previous layer. The message, $R_{j\leftarrow k}^{[l\leftarrow l+1]}$, from the $j$-th neuron of the layer $l+1$ to the $i$-th layer of the layer $l$ is then given by the equation $\ref{eq:message_decomposition}$, where $\Omega_{jk}^{[l+1]}$ is the decomposition rule (see bellow). The new relevance $R_j^{[l]}$ is then obtained by summing each "message" sent, as described by equation $\ref{eq:message_aggregation}$.

$$
\begin{equation}
%\label{eq:message_decomposition}
R_{j\leftarrow k}^{[l\leftarrow l+1]}=\Omega_{jk}^{[l+1]}R_k^{[l+1]}
\end{equation}
$$

$$
\begin{equation}
%\label{eq:message_aggregation}
R_j^{[l]} = \sum_kR_{j\leftarrow k}^{[l\leftarrow l+1]}=\sum_k\Omega_{jk}^{[l+1]}R_k^{[l+1]}
\end{equation}
$$

> [!note] Decomposition Rules
> 
> A rule  $\Omega_{jk}^{[l+1]}$ defines how to redistribute the relevance $R_k^{[l+1]}$ of the layer $l+1$ into the previous layer $l$. The ideal rule depends on the nature of the layer $l+1$ and is an active topic of research [@3@4@5](#resources) (e.g. for new or more complex architecture like Transformers), more on this in the [rule](#different-rules-for-different-layers) and [evaluation](#evaluate-an-explanation) sections.

![layer-wise-relevance-propagation_backward](layer-wise-relevance-propagation_backward.png)
*Figure 1: Relevance back-propagation of the dog logit, [@5](#resources).*
{: .im-center#relevance-back-propagation}

The most basic LRP rule [@1](#resources) is given by the equation $\ref{eq:original_lrp_rule}$ which is a simple pre-activation weighting. One drawback of this rule is that it is not conservative, i.e. $\sum_jR_{j}^{[l]}\neq\sum_kR_{k}^{[l+1]}$ $\Leftrightarrow$ $\sum_j\Omega_{jk}^{[l+1]}\neq1$, and thus relevance get lost or created by the bias along the propagation. Therefore this rule was revisited by [@5](#resources) as LRP-$0$ by setting the biases to $0$ (a trick).

$$
\begin{equation}
%\label{eq:original_lrp_rule}
\Omega_{jk}^{[l+1]}=\dfrac{w_{kj}^{[l+1]}a_{j}^{[l]}}{\sum_{j}w_{kj}^{[l+1]}a_{j}^{[l]}+b_k^{[l+1]}}
\end{equation}
$$

>[!danger] Generalisation Gotcha
>
>The layer superscript is sometimes omitted in the literature because it doesn't cover the general formulation. For example, to account for residual connections the propagation should be spread across multiple layers. The general formulation is thus simply given by a causally weighted sum $R_{j}=\sum_{k}\Omega_{jk}R_k$ where the numbering is across all the neurons (causally means $\Omega_{jk}=0$ if $k<=j$, messages are only sent backwards). It is less informative about the practical implementation but still carries the  idea of tracking the actual computation.

### Different Rules for Different Layers

Intuitively the relevance propagation rules should be dependent on the layers nature as is the forward flow. Fundamentally this idea comes from the ambition of tracking the model's various kind of actual computation into each relevance. The classical framework for deriving these rules are Deep Taylor's series Decomposition (DTD) [@10](#resources), yet it should be justified with care [@6](#resources). I'll now present the necessary rules needed for the experiments I conducted, whose derivation can be found in the various linked [resources](#resources) (most of them assumes ReLU which is the case here).

> [!info] Notation
> 
> The layer superscript is dropped in this section for readability and since all the rules will be applied on consecutive layers. The index $j$ still refers to layer $l$ and the index $k$ to layer $l+1$.

**LRP-$\epsilon$ [@1](#resources):**  Defined by the equation $\ref{eq:lrpe_rule}$, it introduces $\epsilon$, a numerical stabilizer and the $z$ coefficients are set accordingly to the equation $\ref{eq:original_lrp_rule}$ ($z_{jk}=w_{kj}a_j$ and $z_k=\sum_{j}w_{kj}a_{j}+b_k$). A variant of this rule can be computed by setting the biases to $0$ but this is not enough for the propagation to be conservative. The stabiliser $\epsilon$, will absord some relevance and should therefore be kept as small as possible. In practice this rule is made for linear mappings (`Linear` & `BatchNorm`).

$$
\begin{equation}
%\label{eq:lrpe_rule}
\Omega_{jk}=\dfrac{z_{jk}}{z_{k}+\epsilon \cdot {\rm sign}\left(z_{k}\right)}
\end{equation}
$$

**$z^+$ [@10](#resources):** Defined by the equation $\ref{eq:zplus_rule}$, where $w_{kj}^+$ stands for the positive part of $w_{kj}$, i.e. $w_{kj}^+$ if $w_{kj}<0$. This rule is conservative and positive (and thus consistent [@10](#resources)). In practice this rule is used for convolution.

$$
\begin{equation}
%\label{eq:zplus_rule}
\Omega_{jk}=\dfrac{w_{kj}^+a_{j}}{\sum_jw_{kj}^+a_{j}}
\end{equation}
$$

**$w^2$ [@4](#resources):** Defined by the equation $\ref{eq:wsquare_rule}$, this rule is meant for early layers. This rule is conservative and positive (and thus consistent). In practice it is used for the first layer, e.g. in order to propagate relevance to dead input neurons.

$$
\begin{equation}
%\label{eq:wsquare_rule}
\Omega_{jk}=\dfrac{w_{kj}^2}{\sum_j w_{kj}^2}
\end{equation}
$$

**Flat [@4](#resources):** Defined by the equation $\ref{eq:flat_rule}$, this rule is similar to $w^2$ but assuming a constant weighting. It therefore redistributes equally the relevance across every preceding neuron. This rule is conservative and positive (and thus consistent). In practice it is used for the first layer, e.g. in order to propagate relevance to dead input neurons.

$$
\begin{equation}
%\label{eq:flat_rule}
\Omega_{jk}=\dfrac{1}{\sum_j 1}
\end{equation}
$$

### Technical Details

All the LRP computation can be done using the original authors' library [Zennit](https://zennit.readthedocs.io/en/latest/) [@6](#resources).
In practice the graph and back-propagation mechanism of `torch.autograd` is used using backward hooks, see the snipet bellow. The models need to implement the forward pass only using proper modules (child of the model instance) for them to be detected by [Zennit](https://zennit.readthedocs.io/en/latest/) and hooked. And since it relies one the full back-propagation every module of the graph should be hooked (even activation functions).

**Pass rule:** This is a practical rule necessary regarding [Zennit](https://zennit.readthedocs.io/en/latest/) implementation. In practice even activation functions should be hooked because otherwise the classical gradient will be computed during the backward pass. And since the actual relevance propagation is carried by other module hooks (Linear, Conv, etc.) no modification should be done (it's a pass-through). It is typically used for activation functions.

<script src="https://gist.github.com/Xmaster6y/6734100a89f4ab9bd17fe24e84831d40.js"></script>

This important snipet explain how backward hooks are coded in [Zennit](https://zennit.readthedocs.io/en/latest/) which is fundamental to design new rules. Besides **Rules** (derived from the `BasicHook`) other fundamental objects are available:

- **Stabilizers**: $\epsilon$ term in the LRP-$\epsilon$, which make the computation numerically stable. All rules use it in practice.
- `Canonizer`: Needed when using `BatchNorm` layers [@3](#resources), to reorder the computation.
- **Composites**: To easily combine rules.

To illustrate rules below is a snipet of the $z^+$ rule implementation.  weights modifiers LRP-$0$.The input modifiers are how to practically derive the rules with the lightest hook as possible.

<script src="https://gist.github.com/Xmaster6y/a87cb4c058c47558e0f3cb9634b419e5.js"></script>

## Interpreting Othello Zero

### Playing Othello

Before digging into the actual interpretation of the network I borrowed from [Alpha Zero General](https://github.com/suragnair/alpha-zero-general) [@7](#resources), it is important to understand how it is used in practice and how it was trained. I highly recommend to check their code on [Github](https://github.com/suragnair/alpha-zero-general) or the associated [blog post](https://web.stanford.edu/~surag/posts/alphazero.html) as well as the original Alpha Zero paper [@9](#resources).

Tree representation of game (Min-Max, Alpha-Beta, MCTS, ...) is an intuitive representation of a game starting from the root (the current position), the nodes (board states $s$) and the edges (action chosen for a given state $(s,a)$). The Alpha Zero paper [@8](#resources) used MCTS PUCT, with the upper bound confidence (UCB) is given by the equation $\ref{eq:upper_confidence_boundary}$. This equation involves network predictions as $P(s,a)$ is the policy vector and $Q(s,a)$ is the average expected value over the visited children.

$$
\begin{equation}
%\label{eq:upper_confidence_boundary}
    U_s=Q_s+c_{\rm puct}\cdot \pi(s) \cdot \dfrac{\sqrt{||N_{sb}||_1}}{1+N_{s}}
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

It also was a critic of LRP with the DTD framing [@6](#resources) as interpreting an input-dependant heatmap is about interpreting an input and not really the model.

Some faithfullness measure algorithm exisits. I'll describ randomisation using most-relevant pixel flipping. In practice this can be used to compare XAI methods. For example it could be used to verify that the rules derived using DTD are actually the best fited for each layer kind.

## Resources

A drafty notebook that self-contains all the practical experiments presented here and more is available on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ozMKtcRS9nRtvUfwZwj00ZZNpui5MhLr?usp=sharing). And bellow is a list of references containing the papers and code used in this post as well as additional resources.

- [@5](#resources) extends LRP to discover concepts.

> [!quote] References
> 
> 1. Bach, Sebastian, et al. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." _PLOS ONE_, vol. 10, no. 7, 2015.
> 2. Shrikumar, Avanti, et al. "Not Just a Black Box: Learning Important Features Through Propagating Activation Differences." _ArXiv_, 2016.
> 3. Binder, Alexander, et al. "Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers." _ArXiv_, 2016.
> 4. Lapuschkin, Sebastian, et al. "Unmasking Clever Hans Predictors and Assessing What Machines Really Learn." _Nature Communications_, vol. 10, no. 1, 2019.
> 5. Achtibat, Reduan, et al. "From Attribution Maps to Human-understandable Explanations through Concept Relevance Propagation." _Nature Machine Intelligence_, vol. 5, no. 9, 2023.
> 6. Sixt, Leon, and Tim Landgraf. "A Rigorous Study Of The Deep Taylor Decomposition." _ArXiv_, 2022.
> 7. Anders, Christopher J., et al. "Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy." _ArXiv_, 2021.
> 8. Thakoor, Shantanu, et al. "Learning to play othello without human knowledge." _Stanford University_, 2016.
> 9. Silver, David, et al. "Mastering the Game of Go Without Human Knowledge." Nature, vol. 550, no. 7676, 2017.
> 10. Montavon, Grégoire, et al. "Explaining Nonlinear Classification Decisions with Deep Taylor Decomposition." _Pattern Recognition_, vol. 65, 2017.
