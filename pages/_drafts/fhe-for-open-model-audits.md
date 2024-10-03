---
title: FHE for Open Model Audits
tldr: Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, as many, that these advancement are a real opportunity to improve AI safety. I thus outline possible applications in model evaluation and interpretability, the most mature tools in safety in my opinion.
tags:
  - AIS
  - XAI
  - FHE
  - Eval
references: 
aliases: 
crossposts: 
publishedOn: 
editedOn: 
authors:
  - "[[Yoann Poupart]]"
readingTime: 7
image: /assets/images/fhe-for-open-model-audits_thumbnail.webp
description: TL;DR> Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, as many, that these advancement are a real opportunity to improve AI safety. I thus outline possible applications in model evaluation and interpretability, the most mature tools in safety in my opinion.
---

![TFHE for Open Interpretability Audits](fhe-for-open-model-audits.webp)

> [!tldr] TL;DR
> 
> Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, as many, that these advancement are a real opportunity to improve AI safety. I thus outline possible applications in model evaluation and interpretability, the most mature tools in safety in my opinion.

> [!example] Table of content
> 
> - [Context](#context)
> - [What is FHE?](#what-is-fhe)
> 	- [The FHE scheme](#the-fhe-scheme)
> 	- [FHE Applied to ML](#fhe-applied-to-ml)
> - [Hiding the Test Set in Public](#hiding-the-test-set-in-public)
> 	- [The Test Set Game](#the-test-set-game)
> 	- [Zero Trust for Better Safety](#zero-trust-for-better-safety)
> 	- [Computing Private Metrics in Public](#computing-private-metrics-in-public)
> - [Privately Inspecting Model Biases](#privately-inspecting-model-biases)
> 	- [I/O Interpretability Methods](#io-interpretability-methods)
> 	- [What's Missing?](#whats-missing)
> - [Resources](#resources)

## Context

I recently participated in the Privacy Preserving AI Hackathon organised by [Entrepreneur First](https://www.linkedin.com/company/entrepreneur-first/), [Hugging Face](https://www.linkedin.com/company/huggingface/) and [Zama](https://www.linkedin.com/company/zama-ai/). With my team we focused on privately matching patients with clinic trials that have specific requirements. Even though we didn't win I learned a lot, and I want to continue building with this amazing technology!

> [!caution] Epistemic Status
> 
> I am still beginner in TFHE and definitely not a crypto-guy. I know the basics and really love the field, The Code Book [@1](#resources) inspired me to focus on cybersecurity back in  my Bachelor. In comparison I would say that I am well versed on the subject of AI evaluation and interpretability. The following is thus an article from an AI-guy trying to apply TFHE for improving AI safety.

## What is FHE?

### The FHE scheme

First, FHE stands for "fully homomorphic encryption", a framework for cryptosystems supporting computation on cyphertexts. As the name "fully homomorphic" suggest the beauty of the method is to enable a correspondance between computation in the encrypted space and the clear space. Indeed, if $\varepsilon$ is the encryption mechanism, $\varepsilon(\lambda \cdot A + B)=\lambda\circ\varepsilon(A)\star\varepsilon(B)$ such that a series of computation can be performed while encrypted, only needing to decrypt the result.

**Why does it matter?** It matters because it means that the computation can be performed by an untrusted entity w.r.t. the clear inputs. For the entity performing the computation it also means that you don't need to give away your "secret sauce", i.e. the circuit of computation. This scheme adheres to the web3 spirit of "zero trust by design".

- In practice rivacy preserving applications

> [!todo] Privacy Checklist
> 
> - Who wants to protect what?
> - What is encrypted?
> - Where is the compute executed?
> - Who decrypts?
> - Who needs the compute results?
> -  Is FHE really necessary here?
> - Is FHE realistic here?

The last two points are crucial, as you want to avoid using FHE as much as possible. Bear in mind that computing on cyphertexts is expensive! For example if you can run locally on th
It can often be reducted to Is there enough local power to run the model? 

### FHE Applied to ML

In order to be scalable, TFHE [@2](#references) was developed as a fast way to do FHE by Zama.

This mostly enabled to overcom.   Non linear FHE. 

- Learning with error
Fast bootstraping [@3](#references)

> [!info] Going Deeper
> 
> If you want to go deeper and learn about the actual algorithms implemented have a look at [Zama's blog](https://www.zama.ai). They have great resources, tutorials and stories for crypto-experts but also less technical wanderer. I especially recommend their [101](https://www.zama.ai/post/homomorphic-encryption-101) on FHE. 

Under the hood [`conrete-ml`](https://docs.zama.ai/concrete-ml) uses method of universality. Thus you can manipulate and customise your torch model at will providing that you use supported operations.

> [!tip] Universal ML
> 
> As softwares/languages/libraries go there is a need to abstract and universalise certain concepts. As such ONNX is commonly used to share or convert models, e.g. from Tensorflow to PyTorch. On the other side MLIR is used by vendors as a middle-layer between their hardware and high-level programming, natively integrated in new ML-oriented languages like MOJO, worth checking!

In privacy preserving ML applications, the common scheme is often based on:

- A model owner who wants to keep its secret sauce (mostly the model)
- A data owner who wants to keep its privacy

Application example of such a scheme could be a method to classify people based on their private data (bank loans, insurance policies, etc.) or to enable a feature for known users (facial recognition, ADN testing, etc.). 

I argue that AI safety share the same goals as described above. You want to be sure that all models are safe, not only the open source ones. You might want to regulate or inspect models without letting the model owner know, as in many other domains. The current state of affairs is not satisfying and I describe in the next sections what directions could be net beneficial for AI safety.

## Hiding the Test Set in Public

### The Test Set Game

In ML it is well known that you should always keep a subset of data hidden from the model, the test set. Consequently, this dataset provide a faithful model evaluation, should it be big enough  and randomly sampled from the original dataset. It should be not mistaken for the validation set, used for hyperparameters search, which doesn't provide a good evaluation metric due to the Goodhart's law similarily to the train set.

What if you train on the test set? In frugal ML, per say using `scikit-learn`, it only happens willingly in scarce data problems like for health predictions. Except from this particular case, should you encounter a data leakage problem, you often can re-process the datasets and re-train from scratch. In the era of big pre-trained models, such a solution would be far too expensive. Yet, it is a real problem, well illustrated by this satirical article [@4](#resources).

It is a major problem since they scrape over and over the web. You can also forget using machine unlearing for such a scale and diversity of data, you'll dramatically hurt the performances. The two best SOTA approaches are: 

- Don't publish the test set.
- Contaminate the test set.

First it's harder to trust, but the pineacle of. Since the evaluations are send in clear to the models model owners could be saving them. As for the contamination could be overcome by additional training, adversarial attacks or chirurgical machine unlearning.

### Zero Trust for Better Safety

I won't trust model owners with my evaluation data and model owners won't trust my evaluations.



- You'll still need to trust the data provider. But as you make more actors involved you increase trust
	- Data could be approbated by the legislator and used by everyone


- Taking back the control over your own data
- Legal enforcement is not enough
	- Some company are willing to pay fines

### Computing Private Metrics in Public

- While the computing mechanism is clear and public only the decryption key is hidden


## Privately Inspecting Model Biases

### I/O Interpretability Methods

- Perturbation methods
- Surrogate model
- Feature importance

### What's Missing?

- Improve model speed -> Zama working on it
- Improve model support -> If each comp would improve support for its own model could be speed up
- Giving intermediate representations
	- Might be complicated
- 

## Resources

Bellow is a short list of the references cited throughout this blog post.

> [!quote] References
> 
> 1. Singh, Simon. "The Code Book: The Evolution of Secrecy from Mary, Queen of Scots, to Quantum Cryptography" (1st. ed.). Doubleday, USA (1999).
> 2. Chillotti, Ilaria et al. "TFHE: Fast Fully Homomorphic Encryption over the Torus." (2016).
> 3. Stoian, Andrei et al. "Deep Neural Networks for Encrypted Inference with TFHE." (2023)
> 4. Schaeffer, Rylan. "Pretraining on the test set is all you need." _arXiv preprint arXiv:2309.08632_ (2023).
