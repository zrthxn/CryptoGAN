# Notes [Coutinho Implementation]

These are some notes from my implementation of the following papers and experiments.
This will help me keep a record of the things I have done and have to do.

### Observations

- BCE Loss gives better results than L1
- HardSigmoid in last layer

### Things I've tried

- Changing the batch size, [200, 2000, 4000]
  - Batches above 2000 give diminishing returns
  - Below 1000 network doesn't train well
- Smoothing in the trends
  - Using an average rise and fall rate makes graphs better

### Things to Try

- Introduce weights to losses, B and G
- Alter the model size and shape
- Use different type of encoding
