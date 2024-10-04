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
publishedOn: 2024-10-04
editedOn: 2024-10-04
authors:
  - "[[Yoann Poupart]]"
readingTime: 12
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

**Why does it matter?** It matters because it means that the computation can be performed by an untrusted entity w.r.t. the clear inputs. For the entity performing the computation it also means that you don't need to give away your "secret sauce", i.e. the series of computation. This scheme adheres to the spirit of zero trust by design. This enable to build privacy preserving applications that might leverage highly tailored features based on your data without any leak.

> [!todo] Privacy Checklist
> 
> Here is a simple checklist for your privacy preserving application:
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
Fast bootstraping [@3](#references) . Zama improved in the learning with error framework  [@4](#references)

> [!info] Going Deeper
> 
> If you want to go deeper and learn about the actual algorithms implemented have a look at [Zama's blog](https://www.zama.ai). They have great resources, tutorials and stories for crypto-experts but also less technical wanderer. I especially recommend their [101](https://www.zama.ai/post/homomorphic-encryption-101) on FHE. 

The main limitations concerns the data, the model and the computation (so everything). The computation needs to be supported, done with integers and are slow. Under the hood [`conrete-ml`](https://docs.zama.ai/concrete-ml) uses method of universality. Thus you can manipulate and customise your torch model at will providing that you use supported operations. A key aspect of that is that you don't need to train your model with FHE, you only need to compile it, i.e. translate its operations into FHE-friendly operations. [`conrete-ml`](https://docs.zama.ai/concrete-ml) provide native APIs to create an FHE model from regular `torch` or `sklearn`model. They made it so easy, it would be a shame not to use it!

> [!tip] Universal ML
> 
> As softwares/languages/libraries go there is a need to abstract and universalise certain concepts. As such ONNX is commonly used to share or convert models, e.g. from Tensorflow to PyTorch. On the other side MLIR is used by vendors as a middle-layer between their hardware and high-level programming, natively integrated in new ML-oriented languages like MOJO, worth checking!

In privacy preserving ML applications, the common scheme is often based on:

- A model owner who wants to keep its secret sauce (mostly the model).
- A data owner who wants to keep its privacy.

Application example of such a scheme could be a method to classify people based on their private data (bank loans, insurance policies, etc.) or to enable a feature for known users (facial recognition, ADN testing, etc.). In the end what matters is that the user has the same service but takes back the control over its own data

I argue that AI safety share the same goals as described above. You want to be sure that all models are safe, not only the open source ones. You might want to regulate or inspect models without letting the model owner know, as in many other domains. The current state of affairs is not satisfying and I describe in the next sections what directions could be net beneficial for AI safety.

## Hiding the Test Set in Public

### The Test Set Game

In ML it is well known that you should always keep a subset of data hidden from the model, the test set. Consequently, this dataset provide a faithful model evaluation, should it be big enough  and randomly sampled from the original dataset. It should be not mistaken for the validation set, used for hyperparameters search, which doesn't provide a good evaluation metric due to the Goodhart's law similarily to the train set.

What if you train on the test set? In frugal ML, per say using `scikit-learn`, it only happens willingly in scarce data problems like for health predictions. Except from this particular case, should you encounter a data leakage problem, you often can re-process the datasets and re-train from scratch. In the era of big pre-trained models, such a solution would be far too expensive. Yet, it is a real problem, well illustrated by this satirical article [@5](#resources).

It is a major problem since they scrape over and over the web. You can also forget using machine unlearing for such a scale and diversity of data, you'll dramatically hurt the performances. The two best SOTA approaches are: 

- Don't publish the test set.
- Contaminate the test set.

First it's harder to trust, but the pineacle of. Since the evaluations are send in clear to the models model owners could be saving them. As for the contamination could be overcome by additional training, adversarial attacks or chirurgical machine unlearning.

### Zero Trust for Better Safety

I won't trust model owners with my evaluation data and model owners won't trust my evaluations. For this scheme I first provide an encrypted test set and then I run the evaluation publicly, e.g. on an HF space like MTEB. 

We can push the distrust scheme even further. What if model owners didn't wanted to leak on which inputs their model made a mistake? We would process similarly but using a common public key with a segmented private key. First the model owners would provide the encrypted output, impossible to decrypt without their agreement. Then, we would run a circuit computing the accuracy. And finally, together, we would decrypt the metric result. In order to illustrate this process we can simply write a torch model computing the accuracy.

<script src="https://gist.github.com/Xmaster6y/0b84195df1ba6204e6b5b5c88e621472.js"></script>

> [!error] Gotcha
> 
> This problem would require to create an accuracy circuit for each dataset, to generate a new key for each model and to encrypt the test set on-the-fly **every time**. While not practicable it illustrates the power of FHE to operate in scenarios where there is no trust.

**What kind of governance would it enable?** The last detail I previously didn't mention for the sake of illustration is about trusting the data. In order to overcome this issue we'll need an additional entity, a regulator, that would in the end own the data. As such they would be the one to approve the dataset and deliver the associated certificate. The business organisation would be based on:

- A law enforcing requiring model evaluation certificates.
- An evaluator entity, entitled by the legislator to deliver certificates.
- A model company paying the evaluator to abide by the law and look good with the certificate.

Obviously legal enforcement might not be enough. Indeed, some companies are willing to pay fines, even more when benefits largely outweighs the fines. Yet, pushing technologies like FHE might enable new use cases for AI safety.

## Privately Inspecting Model Biases

Can we say more about a model than just its performances with only a limited access?

### I/O Interpretability Methods

In the era of sparse autoencoders or activation manipulation, heavily relying on model internals, let's shade some light on interpretability methods only requiring an input/output mapping, widely used by the non-DL folks.

, the first naive approach would be to recreate it. Depending on the model, e.g. ChatGPT, it might be too expensive but still a good start. In the field of XAI  using a surrogate model with LIME [@6](#resources).

Another very common framework to explain models with only a black box access is the Shapley Values [@7](#resources). Put simply the Shapley Values, originating from game theory, disantangle the contribution of a each player to a common team reward. In XAI we might want to disantangle the contribution from each input feature (player) to the model prediction (reward). Feature importance with SHAP [@8](#resources).

Lastly it is possible to create explanations from mere input data,. What happens if I remove occlusion [@9](#resources) e.g. or anchors [@10](#resources). Perturbation methods with Anchors. unified as "Explaining by Removing" methods [@11](#resources)
More sophisticated methods could make use of adversarial examples. A great example could be using semantically equivalent adversaries [@12](#resources). In general finding adversarial examples is possible, even with a black box access training a surrogate model [@13](#resources).

> [!tip] Transferability
> 
> In a feature world adversairal transferability is no surprise [@14](#resources). Still an object of study, especially for the latest models [@15](#resources). Natural abstraction hypothesis state that dominant concepts will eventually be learned (with enough time and compute) in spite of the learning methodolodgy.

In conclusion there exists a vast pannel of methods to inspect a model only based on I/O. They would provide a deeper understanding of the model, e.g. biases, clever hans or condensed knowledge, while being FHE friendly.

### What's Missing?

Modern interpretability methods studying deep neural networks make use of the models internals like activations, embeddings, individual neurons, weights, gradients, etc. Yet this data is way more sensitive for the model owner than its output. 

Interpretability is more costly than basic evaluation so speed is primordial to improve. Zama working on it as it is a global limitation of FHE.

More generally improve model support -> If each comp would improve support for its own model could be speed up. It would widden adoption.

## Resources

Bellow is a short list of the references cited throughout this blog post.

> [!quote] References
> 
> 1. Singh, Simon. "The Code Book: The Evolution of Secrecy from Mary, Queen of Scots, to Quantum Cryptography" (1st. ed.). Doubleday, USA (1999).
> 2. Chillotti, Ilaria et al. "TFHE: Fast Fully Homomorphic Encryption over the Torus." (2016).
> 3. Chillotti, Ilaria et al. "Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks." (2021).
> 4. Stoian, Andrei et al. "Deep Neural Networks for Encrypted Inference with TFHE." (2023).
> 5. Schaeffer, Rylan. "Pretraining on the test set is all you need." _arXiv preprint arXiv:2309.08632_ (2023).
> 6. Ribeiro, Marco Tulio et al. ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ (2016).
> 7. Shapley, Lloyd S. "Notes on the n-Person Game -- II: The Value of an n-Person Game." Santa Monica, Calif.: RAND Corporation (1951).
> 8. Lundberg, Scott M. and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." _Neural Information Processing Systems_ (2017).
> 9. Zeiler, Matthew D. and Rob Fergus. "Visualizing and Understanding Convolutional Networks." _ArXiv_ abs/1311.2901 (2013).
> 10. Ribeiro, Marco Tulio et al. "Anchors: High-Precision Model-Agnostic Explanations." _AAAI Conference on Artificial Intelligence_ (2018).
> 11. Covert, Ian et al. “Explaining by Removing: A Unified Framework for Model Explanation.” _J. Mach. Learn. Res._ 22 (2020): 209:1-209:90.
> 12. Ribeiro, Marco Tulio et al. "Semantically Equivalent Adversarial Rules for Debugging NLP models." _Annual Meeting of the Association for Computational Linguistics_ (2018).
> 13. Shi, Yi et al. “Active Deep Learning Attacks under Strict Rate Limitations for Online API Calls.” _2018 IEEE International Symposium on Technologies for Homeland Security (HST)_ (2018).
> 14. Ilyas, Andrew et al. "Adversarial Examples Are Not Bugs, They Are Features." _Neural Information Processing Systems_ (2019).
> 15. Schlarmann, Christian and Matthias Hein. "On the Adversarial Robustness of Multi-Modal Foundation Models." _2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)_ (2023).
