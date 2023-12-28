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

## Classification Example

### Network Decomposition

### Interpretation

> [!quote] References
> 
> 1. [[On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation]]
> 2. [[A Rigorous Study Of The Deep Taylor Decomposition]]
