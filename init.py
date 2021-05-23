import logging
import random
import string
from sys import argv
from config import build_config, defaults

from src import trainer, test


def main():
  actions = list()
  for i, arg in enumerate(argv[1:]):
    if arg.find("=") == -1:
      actions.append(arg)
      argv.pop(i)
    else:
      continue

  build_config(argv[1:])
  logging.basicConfig(level=logging.INFO)

  if actions.__contains__("help"):
    print("Training utility")
    return 0

  if actions.__contains__("train"):      
    trainer.start(model=defaults["model"])

  if actions.__contains__("eval"):      
    test.evaluate()

  if actions.__contains__("encrypt"):
    K = ''.join(random.choices(string.ascii_uppercase + string.digits, 
      k = defaults[defaults["model"]]["blocksize"]
    ))
    C = test.encrypt(input("Text: "), K)
  if actions.__contains__("decrypt"):      
    P = test.decrypt(C, K)
    print("Decrypted:", test.decode(P))
  
if __name__ == "__main__":
  main()
