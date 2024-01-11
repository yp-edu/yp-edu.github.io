---
title: Layer-Wise Relevance Propagation
tldr: 
tags: 
references: 
aliases: 
crossposts: 
publishedOn: 
editedOn: 
authors:
  - "[[Yoann Poupart]]"
readingTime:
---
> [!caution] WIP
> 
> This article is a work in progress.

> [!tldr] TL;DR
> 
> LRP is a method that produces pixel relevances for a given output which doesn't to be terminal. Technically the computation happens using a single back-progation pass. 

> [!example] Table of content
> 
> - [LRP Framework](#lrp-framework)
> 	- [Formulation](#formulation)
> 	-  [Different Rules](#different-rules)
> 	-  [Technical Details](#technical-details)
> - [Classification Example](#classification-example)
> 	- [Network Decomposition](#network-decomposition)
> 	- [Interpretation](#interpretation)

## LRP Framework

### Formulations

With $R_j^{[l]}$ being the $j$-th neuron's relevance of the layer $l$, and the propagation mechanism is given by the equation $\ref{eq:aggregate}$.

$$
\begin{equation}
%\label{eq:aggregate}
R_{j}^{[l]}=\sum_{k}\dfrac{z_{jk}}{\sum_j z_{kj}}R_k^{[l+1]}
\end{equation}
$$

### Different Rules

### Technical Details


## Interpreting Othello Zero

### Game

{% gist fd8ff108d39b0fdd09cb49e6809d2c54 %}

### Network Decomposition

<script src="https://gist.github.com/Xmaster6y/fd8ff108d39b0fdd09cb49e6809d2c54.js"></script>

### Interpretation

> [!quote] References
> 
> 1. Bach, Sebastian, et al. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." _PLOS ONE_, vol. 10, no. 7, 2015, p. e0130140, https://doi.org/10.1371/journal.pone.0130140.
> 2. Sixt, Leon, and Tim Landgraf. "A Rigorous Study Of The Deep Taylor Decomposition." _ArXiv_, 2022, /abs/2211.08425.
> 3. Binder, Alexander, et al. "Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers." _ArXiv_, 2016, /abs/1604.00825.