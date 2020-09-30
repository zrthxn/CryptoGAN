# Adversarial Neural Cryptography
A GAN based approach to break, inject and/or forge communication between two symmetrically secure parties.

### Topics to Explore

Programming an adversarial network
- PyTorch implementation
- train to perform basic task
- network against copy of itself
- theory and practice

Model a network like an encryptor
- architecture of net like crypto function
- measure loss against actual function
- is the net able to reliably encrypt?
- can a decryptor net be made to do the reverse?
- model a decryptor net
- what's avg data loss

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