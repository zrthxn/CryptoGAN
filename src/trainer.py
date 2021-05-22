from os import system
from torch import save as save_model
from datetime import datetime

from config import defaults
from src.anc import training as ancTrainer
from src.cryptonet import training as cryptonetTrainer


def start(model, debug = defaults["debug"]):
  if not model:
    raise ValueError("Please provide a model name to train.")

  if model not in defaults["avail_models"]:
    raise NameError("This model is not available.")


  if model == "anc":
    # ANC Training
    session = ancTrainer.TrainingSession(debug=debug)
  elif model == "cryptonet":
    # Cryptonet Training 
    session = cryptonetTrainer.TrainingSession(debug=debug)

  # if defaults["tensorboard"]:
  #   system(f"tensorboard --logdir {session.logdir} &")
  trained = session.train(BATCHES=defaults["training"]["batches"], EPOCHS=defaults["training"]["epochs"])
  models, losses = trained

  for trained in models:
    if defaults["save_model"]:
      save_model(trained.state_dict(), f'models/{model}/{trained.name}_{datetime.now()}.mdl')

  return losses
