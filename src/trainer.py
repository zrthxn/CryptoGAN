from anc import training as ancTraining
from cryptonet import training as cryptonetTraining

BLOCKSIZE = 12
BATCHLEN = 64

BATCHES = 256
EPOCHS = 10

session = ancTraining.TrainingSession(debug=True, BLOCKSIZE=BLOCKSIZE, BATCHLEN=BATCHLEN)
trained = session.train(BATCHES=BATCHES, EPOCHS=EPOCHS)

# session = cryptonetTraining.TrainingSession(BLOCKSIZE, BATCHLEN)
# trained = session.train(BATCHES, EPOCHS)
