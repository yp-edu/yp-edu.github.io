---
title: Evidence of Learned Look-Ahead
tldr: Reproduction of the paper Evidence of Learned Look-Ahead using my own interpretability library lczerolens. This article investigate how chess agents can predict
tags:
  - XAI
  - AlphaZero
  - MI
references: 
aliases: 
crossposts: 
publishedOn: 2024-06-27
editedOn: 2024-06-27
authors:
  - "[[Yoann Poupart]]"
readingTime: 18
image: /assets/images/evidence-of-learned-look-ahead_thumbnail.png
description: TL;DR> Layer-Wise Relevance Propagation (LRP) is a propagation method that produces relevances for a given input with regard to a target output. Technically the computation happens using a single back-progation pass similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.
---

![Evidence of Learned Look-Ahead](evidence-of-learned-look-ahead.png)

> [!tldr] TL;DR
> 
> Layer-Wise Relevance Propagation (LRP) is a propagation method that produces relevances for a given input with regard to a target output. Technically, the computation happens using a single back-propagation pass, similarly to deconvolution. I propose to illustrate this method on an Alpha-Zero network trained to play Othello.

> [!example] Table of content
> 
> - [Background](#background)
> 	- [Chess-Playing Agents](#chess-playing-agents)
> - [Resources](#resources)

## Background

### Chess-Playing Agents

### Probing

### Activation Patching

## Probing Results

## Activation Patching Results

## Resources

A 

> [!quote] References
> 
> 1. Bach, Sebastian, et al. "On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation." _PLOS ONE_, vol. 10, no. 7, 2015.

