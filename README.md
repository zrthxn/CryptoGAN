# CryptoGAN

[![Gitpod ready-to-code](https://img.shields.io/badge/Gitpod-ready--to--code-blue?logo=gitpod)](https://gitpod.io/#https://github.com/zrthxn/CryptoGAN)
![License](https://img.shields.io/github/license/zrthxn/cryptogan)
![Last commit](https://img.shields.io/github/last-commit/zrthxn/cryptogan)

A GAN based approach to encrypts communication between two symmetrically secure parties.
Based on the seminal paper by Abadi et al (2017) on [Adversarial Neural Cryptography](https://arxiv.org/pdf/1610.06918.pdf).

This project is an attempt to implement the concepts laid out in the literature and produce results that indicate the efficacy of neural cryptography.

<img src="./docs/ref/anc.png" width="40%">
<img src="./docs/ref/anclayers.png" width="40%">

### Abstract
In this project, we demonstrate that neural networks can learn to protect communications, 
and build a network which can encrypt and decrypt bit-strings.
The learning does not require prescribing a particular set of cryptographic algorithms, 
nor indicating ways of applying these algorithms. We do not prescribe specific cryptographic 
algorithms to these neural networks; instead, we train end-to-end, adversarially. 

## Status
The project is being developed with a focus on finding a more effective strategy and use-case.
The most viable field of application seems to currently be in natual language. 