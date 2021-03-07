from torch import save as save_model
from datetime import datetime

from anc import training as ancTrainer
from cryptonet import training as cryptonetTrainer

debug = False


# ANC Training
session = ancTrainer.TrainingSession(debug, BLOCKSIZE=16, BATCHLEN=2048)
trained = session.train(BATCHES=15000, EPOCHS=10)

models, losses = trained

for model in models:
  save_model(model.state_dict(), f'models/anc/{model.name}_{datetime.now()}.mdl')


# # Cryptonet Training 
# session = cryptonetTrainer.TrainingSession(debug, BLOCKSIZE=16, BATCHLEN=16)
# trained = session.train(BATCHES=24, EPOCHS=1)

# models, losses = trained

# for model in models:
#   save_model(model.state_dict(), f'models/cryptonet/{model.name}_{datetime.now()}.mdl')

