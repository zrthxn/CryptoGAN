from cryptonet import training

session = training.TrainingSession()

alice_loss, bob_loss, eve_loss = session.train()
