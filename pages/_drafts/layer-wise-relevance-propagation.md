---
title: Layer-Wise Relevance Propagation
tldr: LRP is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
tags:
  - XAI
references: 
aliases: 
crossposts: 
publishedOn: 2024-01-12
editedOn: 2024-01-12
authors:
  - "[[Yoann Poupart]]"
readingTime: 
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
> 	-  [Different Rules](#different-rules)
> 	-  [Technical Details](#technical-details)
> - [Classification Example](#classification-example)
> 	- [Network Decomposition](#network-decomposition)
> 	- [Interpretation](#interpretation)
> - [Resources](#resources)

## LRP Framework

### Formulation

Each neuron of each layer can be interpreted. Here a neuron is understood input specific as for a linear layer the interpreted neuron would be for $a(w^Tx+b)$.

![layer-wise-relevance-propagation_backward](layer-wise-relevance-propagation_backward.png)
*Relevance Backpropagation of the dog logit, [@7](#resources).*
{: .im-center}


The general propagation mechanism is given by the equation $\ref{eq:aggregate}$, with $R_j^{[l]}$ being the $j$-th neuron's relevance of the layer $l$, and $z_k=\sum_jz_{kj}$ is the normalisation factor.

$$
\begin{equation}
%\label{eq:aggregate}
R_{j}^{[l]}=\sum_{k}\dfrac{z_{jk}}{z_{k}+\epsilon \sign(z_k)}R_k^{[l+1]}
\end{equation}
$$

The coefficients $z_{jk}$ define how the information is propagated. The term in $\epsilon$ is a numerical stabilizer but has the drawback to make the propagation mechanism not conservative.

Bias is also absorbing the relevance along the way, being a leaf in the the computational graph.

### Different Rules for Differen Layers

- Epsilon
- Zplus
- Pass

### Technical Details

- Backpass modification
- Backward hooks
- Stabilisers
- Cannonisers
- Input modifiers
- Weights modifiers

## Interpreting Othello Zero

### Game

- Alpha Zero
- MCTS PUCT
- UCB
- 

<script src="https://gist.github.com/Xmaster6y/fd8ff108d39b0fdd09cb49e6809d2c54.js"></script>
### Network Decomposition

In order to use Znnit it is important to remember how it is implemented. Many difficulties arise in practice.

First all used modules should be instanciated under the target module (even activations). Then softmax should not be used because of the exponential. Here it can be safely removed as the output are the soflogmax which is a simple translation of the raw logits and doesn't change the softmax used after to select the next action.

It is important to acknowledge the similarity in computation of the value and the policy. This will lead to very close relevances heatmap as only one layer differ. 

One other practical limitation concern the empty cells. Using traditional LRP rules their relevance will be zero. Indeed during the computation the model doesn't use these pixels but it rather uses biases relevances. In order to overcome this difficulty it is possible to use a Flat rule, equation $\ref{eq:flat_rule}$, in the first layer to distribute equally the relevances among pixels.

$$
\begin{equation}
%\label{eq:flat_rule}
R_{j}^{[l]}=\sum_{k}\dfrac{1}{\sum_j 1}R_k^{[l+1]}
\end{equation}
$$
### Interpretation

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
> 2. Sixt, Leon, and Tim Landgraf. "A Rigorous Study Of The Deep Taylor Decomposition." _ArXiv_, 2022.
> 3. Binder, Alexander, et al. "Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers." _ArXiv_, 2016.
> 4. Lapuschkin, Sebastian, et al. "Unmasking Clever Hans Predictors and Assessing What Machines Really Learn." _Nature Communications_, vol. 10, no. 1, 2019.
> 5. Thakoor, Shantanu, et al. "Learning to play othello without human knowledge." _Stanford University_, 2016.
> 6. Anders, Christopher J., et al. "Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy." _ArXiv_, 2021.
> 7. Achtibat, Reduan, et al. "From Attribution Maps to Human-understandable Explanations through Concept Relevance Propagation." _Nature Machine Intelligence_, vol. 5, no. 9, 2023.
