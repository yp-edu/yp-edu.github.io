---
title: Layer-Wise Relevance Propagation
tldr: LRP is a method that produces pixel relevances for a given output which doesn't to be terminal. Technically the computation happens using a single back-progation pass.
tags:
  - XAI
references: 
aliases: 
crossposts: 
publishedOn: 
editedOn: 
authors:
  - "[[Yoann Poupart]]"
readingTime: 
image: /assets/images/lrp_main_image.png
description: The description
---
> [!caution] WIP
> 
> This article is a work in progress.

![post image](lrp_main_image.png)

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
> - [Resources](#resources)

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

<script src="https://gist.github.com/Xmaster6y/fd8ff108d39b0fdd09cb49e6809d2c54.js"></script>
### Network Decomposition

In order 




### Interpretation


![bias_relevance](lrp_bias_relevance.png)
{: .im-center}

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
