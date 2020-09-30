# Notes

These are some notes from my implementation of the following papers and experiments.
This will help me keep a record of the things I have done and have to do.

## 1. Abadi Implementation
[Learning to protect communications with adversarial neural cryptography](https://arxiv.org/pdf/1610.06918.pdf)<br>
Abadi, M.; Andersen, D.G. Learning to protect communications with adversarial neural cryptography. arXiv 2016, arXiv:1610.06918.


  ### Observations
  - Quad loss suggested by paper gives worse results than linear
  - MSE Loss gives better results than L1
  - BCE Loss gives better results than MSE
  - HardSigmoid in last layer instead of tanh, combined with
    BCE loss looks promising


  ### Things I've tried
  - Changing the batch size, [200, 2000, 4000]
    - Batches above 2000 give diminishing returns
    - Below 1000 network doesn't train well
  - Smoothing in the trends
    - Using an average rise and fall rate makes graphs better
  - Using quadratic loss i.e. **{(N/2) - Eve L1}pow2 / (N/2)pow2** as suggested
    - Gives worse results
    - Doesn't make a lot of sense
  - Using *hardtanh* in last layer instead of tanh
   - better results but inconclusive
  - Using *hardsigmoid* in last layer instead of tanh
   - unsatisfactory results with small batches
  - Using BCE Loss with *hardsigmoid* in last layer
   - Better results with opposing bit errors
   - correlation matrix might show that after a point when Bob does well,
     Eve performs poorly and vice-versa.
  - Introducing **weights to losses** for Alice
    - B and G control how much Alice is affected by Bob and Eve respectively
    - With B=2.0 G=1.0, Alice's loss shows high correlation with Bob
    - With B=1.0 G=2.0, Alice's loss shows high correlation with Eve
    - Bit error stagnated around 8 bits which is bad


  ### Things to Try
  - Introduce weights to losses, B and G
  - Alter the model size and shape
  - Use different type of encoding

<hr>

## 2. Coutinho Implementation
### Observations
### Things I've tried
### Things to Try
