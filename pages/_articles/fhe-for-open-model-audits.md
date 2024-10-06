---
title: FHE for Open Model Audits
tldr: Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, like many, that these advancements are a real opportunity to improve AI safety. I thus outline possible applications of FHE in model evaluation and interpretability, the most mature tools in safety as of today in my opinion.
tags:
  - AIS
  - XAI
  - FHE
  - Eval
references: 
aliases: 
crossposts: 
publishedOn: 2024-10-05
editedOn: 2024-10-06
authors:
  - "[[Yoann Poupart]]"
readingTime: 14
image: /assets/images/fhe-for-open-model-audits_thumbnail.webp
description: TL;DR> Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, like many, that these advancements are a real opportunity to improve AI safety. I thus outline possible applications of FHE in model evaluation and interpretability, the most mature tools in safety as of today in my opinion.
---

![TFHE for Open Interpretability Audits](fhe-for-open-model-audits.webp)

> [!tldr] TL;DR
> 
> Thanks to recent developments, FHE can now be applied easily and scalably to deep neural networks. I think, like many, that these advancements are a real opportunity to improve AI safety. I thus outline possible applications of FHE in model evaluation and interpretability, the most mature tools in safety as of today in my opinion.

> [!example] Table of content
> 
> - [Context](#context)
> - [What is FHE?](#what-is-fhe)
> 	- [The FHE scheme](#the-fhe-scheme)
> 	- [FHE Applied to ML](#fhe-applied-to-ml)
> - [Hiding the Test Set in Public](#hiding-the-test-set-in-public)
> 	- [The Test Set Game](#the-test-set-game)
> 	- [Zero Trust for Better Safety](#zero-trust-for-better-safety)
> - [Privately Inspecting Model Biases](#privately-inspecting-model-biases)
> 	- [I/O Interpretability Methods](#io-interpretability-methods)
> 	- [What's Missing?](#whats-missing)
> - [Resources](#resources)

## Context

I recently participated in the Privacy Preserving AI Hackathon organised by [Entrepreneur First](https://www.linkedin.com/company/entrepreneur-first/), [Hugging Face](https://www.linkedin.com/company/huggingface/) and [Zama](https://www.linkedin.com/company/zama-ai/). With my team we focused on privately matching patients with clinic trials that have specific requirements. Even though we didn't win, I learned a lot, and I want to continue building with this amazing technology!

> [!caution] Epistemic Status
> 
> I am still beginner in FHE and definitely not a crypto-guy. I know the basics and really love the field, The Code Book [@1](#resources) inspired me to focus on cybersecurity back in  my Bachelor. In comparison I would say that I am well versed on the subject of AI evaluation and interpretability. The following is thus an article from an AI-guy trying to apply FHE for the sake of improving AI safety.

## What is FHE?

### The FHE scheme

First, FHE stands for "fully homomorphic encryption", a framework for cryptosystems supporting computation on cyphertexts. As the name "fully homomorphic" suggests, the beauty of the method is enabling a correspondence between computation in the encrypted space and the clear space. Indeed, if $\varepsilon$ is the encryption mechanism, $\varepsilon(\lambda \cdot A + B)=\lambda\circ\varepsilon(A)\star\varepsilon(B)$, such that a series of computations can be performed while encrypted, only needing to decrypt the result.

**Why does it matter?** It matters because it means that the computation can be performed by an untrusted entity w.r.t. the input data. For the entity performing the computation, it also means that you don't need to give away your "secret sauce", i.e. the series of computations. This scheme adheres to the spirit of zero trust by design. This enables the building of privacy-preserving applications that might leverage highly tailored features based on your personal data without any leak. Such applications could range from content recommendation, based on your age, sex or localisation, to health checks.

> [!todo] Privacy Checklist
> 
> Here is a simple checklist for your privacy-preserving application:
> 
> - Who wants to protect what?
> - What is encrypted?
> - Where is the compute executed?
> - Who decrypts?
> - Who needs the computed results?
> -  Is FHE really necessary here?
> - Is FHE realistic here?

 Bear in mind that computing on cyphertexts is expensive, so you might want to avoid using FHE as much as possible! For example, you should prefer running the non-critical parts of your system locally, on the client machine, to the extent of the power needed. You'll then keep the encryption scheme for most sensitive business-related computations. 

### FHE Applied to ML

FHE was thought of at the end of the last century and was practically discovered in 2009 [@2](#references), but it still needed to be scalable. Fortunately, the last decade has shown great improvement in the various implementations of the FHE scheme, like with TFHE [@3](#references), enabling efficient addition and multiplication over cyphertexts. What's more, with the adoption and ecosystem growth, private companies like [Zama](https://www.linkedin.com/company/zama-ai/) were created and participated in improving FHE.

What particularly interests us here is how to perform non-linear computation. This step is crucial to handle neural networks' activation functions and other ML algorithms.
This was mostly made possible by fast bootstrapping [@4](#resources), the evolution of the original bootstrapping trick [@2](#references), which was then improved by [Zama](https://www.linkedin.com/company/zama-ai/), who made it scalable [@5](#references).

> [!info] Going Deeper
> 
> If you want to go deeper and learn about the actual algorithms implemented have a look at [Zama's blog](https://www.zama.ai). I especially recommend their [101](https://www.zama.ai/post/homomorphic-encryption-101) on FHE, which clearly explains bootstrapping and its application to neural networks. In general, they have great resources, tutorials, and stories for crypto experts, but also for less technical wanderers.

  In practice, you need to remember that all the computations are done with integers and are slower. With Zama's library [`concrete-ml`](https://docs.zama.ai/concrete-ml), you don't need to handle or write any crypto code. You'll be able to manipulate and customise your `torch` model at will, providing that you use supported operations. Indeed, when you compile your model, it will be translated under the hood into FHE-friendly operations. [`concrete-ml`](https://docs.zama.ai/concrete-ml) provides native APIs to create an FHE model from a regular `torch` or `scikit-learn` model. They made it so easy, it would be a shame not to use it!

> [!tip] Universal ML
> 
> As software/languages/libraries go, there is a need to abstract and universalise certain concepts. As such, ONNX is commonly used to share or convert models, e.g. from Tensorflow to PyTorch. On the other side MLIR is used by vendors as a middle-layer between their hardware and high-level programming, natively integrated in new ML-oriented languages like MOJO, worth checking!

In privacy-preserving ML applications, the common scheme is often based on:

- A model owner who wants to keep its secret sauce (mostly the model).
- A data owner who wants to keep their privacy while accessing a tailored feature.

An application example of such a scheme could be a method to classify people based on their private data (bank loans, insurance policies, etc.) or to enable a feature for known users (facial recognition, DNA testing, etc.). In the end, what matters is that the user has the same service but takes back control over their own data.

I argue that AI safety shares the same goals as described above. You want to be sure that all models are safe, not only the open-source ones. You might want to regulate or inspect models without letting the model owner know, as in many other domains. The current state of affairs could be more satisfying, and I will describe in the next sections what directions could be net beneficial for AI safety.

## Hiding the Test Set in Public

### The Test Set Game

In ML, it is well known that you should always keep a subset of data hidden from the model, the test set. Consequently, this dataset provides a faithful model evaluation, should it be big enough and randomly sampled from the original dataset. It should be not mistaken for the validation set, used for hyperparameters search, which doesn't provide a good evaluation metric due to the Goodhart's law similarily to the train set.

But what if you train on the test set? In frugal ML, per se using `scikit-learn`, it only happens willingly in scarce data problems like health predictions. Except for this particular case, should you encounter a data leakage problem, you can often re-process the datasets and re-train from scratch. Yet, in the era of big pre-trained models, such a solution would be far too expensive. It is thus a real problem, well illustrated by this satirical article [@6](#resources). You can also think of using machine unlearning for such a scale and diversity of data; you would dramatically hurt the performances.

This major problem is primarily due to an aggressive data harvest that scrapes the web and, willingly or not, incorporates a test set of different known benchmarks. In order to avoid or control this leak, the two best SOTA approaches are: 

- Don't publish the test set.
- Contaminate the test set.

Yet, since the evaluations are sent in clear to the models, they could be saved and later be reused. As for the contamination, it could be overcome by additional training, adversarial attacks, or surgical machine unlearning.

> [!success] Policies
> 
> Fortunately, certain model providers, like [Azure](https://azure.microsoft.com), adopted privacy policies stating that no data sent to the model would be collected. This is often encouraged by treaties like the GDPR.

### Zero Trust for Better Safety

It's thus clear that you shouldn't trust model owners with your evaluation data. However, model owners could also mistrust the evaluations. To solve this, I propose a simple scheme. First, provide an encrypted test set and then run the evaluation publicly, e.g., on an HF space like MTEB. 

We can push the distrust scheme even further. What if model owners didn't want to leak on which inputs their model made a mistake? We would process similarly but using a common public key with a segmented private key. First, the model owners would provide the encrypted output, which is impossible to decrypt without their agreement. Then, we would run a circuit computing the accuracy. And finally, together, we would decrypt the metric result. In order to illustrate this process we can simply write a torch model computing the accuracy.

<script src="https://gist.github.com/Xmaster6y/0b84195df1ba6204e6b5b5c88e621472.js"></script>

> [!error] Gotcha
> 
> This problem would require the creation of an accuracy circuit for each dataset, generating a new key for each model and encrypting the test set on the fly **every time**. While not practicable it illustrates the power of FHE to operate in scenarios where there is absolutely no trust.

**What kind of governance would it enable?** The last detail I previously didn't mention for the sake of illustration is about trusting the data. In order to overcome this issue, we'll need an additional entity, a regulator, that would, in the end, own the data. As such, they would be the ones to approve the dataset and deliver the associated certificate. The business organisation would be based on:

- A law enforcing requiring model evaluation certificates.
- An evaluator entity, entitled by the legislator to deliver certificates.
- A model company paying the evaluator to abide by the law and look good with the certificate.

Obviously, legal enforcement might not be enough. Indeed, some companies are willing to pay fines, even more when benefits largely outweighs the fines. Yet, pushing technologies like FHE might enable new use cases for AI safety.

## Privately Inspecting Model Biases

Can we say more about a model than just its performances with only limited access?

### I/O Interpretability Methods

In the era of sparse autoencoders or activation manipulation, heavily relying on model internals, let's shed some light on interpretability methods only requiring input/output mapping, widely used by the non-DL folks.

One of the first naive approaches to analyse a true black-box would be to recreate it. Depending on the model, e.g. ChatGPT, it might be too expensive but still a good start if done locally. In the field of XAI, LIME [@7](#resources) was proposed using a much simpler surrogate model trained locally. Linear models, like logistic regression, are chosen for their straightforward interpretation and can be seen as a local Taylor approximation.

Another very common framework to explain models with only a black box access is the Shapley Values [@8](#resources). Put simply, the Shapley Values, originating from game theory, disentangle the contribution of each player to a common team reward. In XAI we might want to disentangle the contribution from each input feature (player) to the model prediction (reward). Feature importance was efficiently implemented in SHAP [@9](#resources), and remains model agnostic.

Last but not least, it is possible to create explanations centred on input data. Like the intuitive methods of occlusion [@10](#resources), which answers "What happens if I remove this part of the data?", or anchors [@11](#resources), which answers "What are parts of the data must remain to keep the same prediction?". These methods, along with LIME and SHAP, were unified as a set of "Explaining by Removing" methods [@12](#resources). More sophisticated methods could make use of adversarial samples, like using semantically equivalent adversaries [@13](#resources). In general, finding adversarial examples is possible, even with black box access, by training a surrogate model and using transfer [@14](#resources).

> [!tip] Transferability
> 
> Certain adversarial attacks are transferable by nature, as related by meaningful features [@15](#resources). Indeed, the link between model robustness and adversarial attacks is still an active topic, especially for the latest models [@16](#resources). It aligns with the natural abstraction hypothesis, which states that dominant concepts will eventually be learned (with enough time and computation) despite the learning methodology.

In conclusion, there exists a vast panel of methods to inspect a model only based on I/O. They would provide a deeper understanding of the model, e.g. biases, clever hans or condensed knowledge, while being FHE friendly.

### What's Missing?

**Model internals.** Modern post-hoc interpretability methods studying deep neural networks use the model's internals, such as activations, embeddings, individual neurons, weights, gradients, etc. Yet this data is way more sensitive for the model owner than its output since it would drastically lower the replication barrier. It would also be illusory to ask them to compute the interpretability methods themselves, as they could impair the results.

**Speed** As previously said FHE suffer from performance issues compare to clear evaluation. And while evaluation only require one forward pass interpretability often require many, especially in with black-box access. So speed would be primordial to improve, but obviously Zama, and others, are working on it as it would drastically improve FHE usability.

**Model support.** While it is true that `concrete-ml` makes it super easy to rewrite a torch model, an auto-converter would increase development speed. Imagine if you could to compile any model to FHE, HuggingFace would the perfect ecosystem (it already is). More generally improving model support, that could be helped by private companies, would speed and widen adoption.

While it is true that such adjuncts would be terrific, I think that you can already build solid and useful software with it.

## Resources

Below is a short list of the references cited throughout this blog post.

> [!quote] References
> 
> 1. Singh, Simon. "The Code Book: The Evolution of Secrecy from Mary, Queen of Scots, to Quantum Cryptography" (1st. ed.). Doubleday, USA (1999).
> 2. Craig Gentry. "A fully homomorphic encryption scheme." PhD thesis, Stanford University, (2009).
> 3. Chillotti, Ilaria et al. "TFHE: Fast Fully Homomorphic Encryption over the Torus." (2016).
> 4. Chillotti, Ilaria et al. "Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks." (2021).
> 5. Stoian, Andrei et al. "Deep Neural Networks for Encrypted Inference with TFHE." (2023).
> 6. Schaeffer, Rylan. "Pretraining on the test set is all you need." _arXiv preprint arXiv:2309.08632_ (2023).
> 7. Ribeiro, Marco Tulio et al. ""Why Should I Trust You?": Explaining the Predictions of Any Classifier." _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ (2016).
> 8. Shapley, Lloyd S. "Notes on the n-Person Game -- II: The Value of an n-Person Game." Santa Monica, Calif.: RAND Corporation (1951).
> 9. Lundberg, Scott M. and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." _Neural Information Processing Systems_ (2017).
> 10. Zeiler, Matthew D. and Rob Fergus. "Visualizing and Understanding Convolutional Networks." _ArXiv_ abs/1311.2901 (2013).
> 11. Ribeiro, Marco Tulio et al. "Anchors: High-Precision Model-Agnostic Explanations." _AAAI Conference on Artificial Intelligence_ (2018).
> 12. Covert, Ian et al. “Explaining by Removing: A Unified Framework for Model Explanation.” _J. Mach. Learn. Res._ 22 (2020): 209:1-209:90.
> 13. Ribeiro, Marco Tulio et al. "Semantically Equivalent Adversarial Rules for Debugging NLP models." _Annual Meeting of the Association for Computational Linguistics_ (2018).
> 14. Shi, Yi et al. “Active Deep Learning Attacks under Strict Rate Limitations for Online API Calls.” _2018 IEEE International Symposium on Technologies for Homeland Security (HST)_ (2018).
> 15. Ilyas, Andrew et al. "Adversarial Examples Are Not Bugs, They Are Features." _Neural Information Processing Systems_ (2019).
> 16. Schlarmann, Christian and Matthias Hein. "On the Adversarial Robustness of Multi-Modal Foundation Models." _2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)_ (2023).
