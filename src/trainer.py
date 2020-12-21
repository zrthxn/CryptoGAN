from cryptonet import training

session = training.TrainingSession()
trained = session.train(
            BATCHLEN=32,
            BATCHES=256,
            EPOCHS=10
          )
