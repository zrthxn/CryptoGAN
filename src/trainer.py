from anc import training as ancTrainer
# from cryptonet import training as cryptonetTrainer

BLOCKSIZE = 12
BATCHLEN = 64

BATCHES = 256
EPOCHS = 10

session = ancTrainer.TrainingSession(debug=True, BLOCKSIZE=BLOCKSIZE, BATCHLEN=BATCHLEN)
trained = session.train(BATCHES=BATCHES, EPOCHS=EPOCHS)

# session = cryptonetTrainer.TrainingSession(BLOCKSIZE, BATCHLEN)
# trained = session.train(BATCHES, EPOCHS)
