from anc import training as ancTrainer
# from cryptonet import training as cryptonetTrainer

BLOCKSIZE = 16
BATCHLEN = 64

BATCHES = 256
EPOCHS = 1

session = ancTrainer.TrainingSession(BLOCKSIZE=16, BATCHLEN=64)
trained = session.train(BATCHES=256, EPOCHS=1)

models, losses = trained

# session = cryptonetTrainer.TrainingSession(BLOCKSIZE, BATCHLEN)
# trained = session.train(BATCHES, EPOCHS)
