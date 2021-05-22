import logging
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

  if actions.__contains__("logs"):      
    defaults["tensorboard"] = True

  if actions.__contains__("train"):      
    trainer.start(model=defaults["model"])

  if actions.__contains__("eval"):      
    test.evaluate()

  if actions.__contains__("encrypt"):      
    c = test.encrypt("Hello World!1234", "KEYDKEYDKEYDKEYD")
    p = test.decrypt(c, "KEYDKEYDKEYDKEYD")
  
if __name__ == "__main__":
  main()
