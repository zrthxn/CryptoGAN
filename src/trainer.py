from torch import save as save_model
from datetime import datetime

from anc import training as ancTrainer
# from cryptonet import training as cryptonetTrainer

debug = False

session = ancTrainer.TrainingSession(debug, BLOCKSIZE=16, BATCHLEN=16)
# session = cryptonetTrainer.TrainingSession(BLOCKSIZE, BATCHLEN)
trained = session.train(BATCHES=24, EPOCHS=1)

models, losses = trained

for model in models:
  save_model(model.state_dict(), f'models/anc/{model.name}_{datetime.now()}.mdl')

