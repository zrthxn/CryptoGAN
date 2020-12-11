# Notes

These are some notes from my implementation of the following papers and experiments.
This will help me keep a record of the things I have done and have to do.

## Models

The following models were built

1. ANC model - Abadi et al
2. CryptoNet - Coutinho et al

### Observations

- Quad loss suggested by paper gives worse results than linear
- MSE Loss gives better results than L1
- BCE Loss gives better results than MSE
- HardSigmoid in last layer instead of tanh, combined with
  BCE loss looks promising

### Topics to Explore

Programming an adversarial network

- PyTorch implementation
- theory and practice

Model a network like an encryptor

- architecture of net like crypto function
- measure loss against actual function
- is the net able to reliably encrypt?
- can a decryptor net be made to do the reverse?
- model a decryptor net, what's avg data loss

Generative adversarial nets for decryption

- using GAN evesdropper to break security
- verification of paper on learning adversarial security
- can the generator work against another Eve?
- the perfect attacker, what would it be?

One shot learning to build the perfect attacker

- can one shot be used to train an Eve to be perfect?
- how quickly can the Eve be trained?
- can the Eve imitate the generator?
- can the Eve be trained to forge hashes/checksums?
- train an Eve to imitate SHA behaviour? (I think no)
