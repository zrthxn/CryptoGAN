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
