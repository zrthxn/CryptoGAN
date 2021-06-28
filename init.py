import logging
from sys import argv
from config import build_config, defaults

from src import trainer, test
from src.util.binary import str_to_bintensor


def main():
  actions = list()
  for arg in argv[1:]:
    if arg.find("=") == -1:
      actions.append(arg)
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
    P = input("Text: ")
    K = test.keygen()
    C = test.encrypt(P, K)

    if actions.__contains__("decrypt"):
      D, G = test.decrypt(C, K, witheve = actions.__contains__("eve"))
      print("Decrypted:", test.decode(D))
      if actions.__contains__("eve"): 
        print("Eve Guess:", test.decode(G))
      test.evaluate_manual(P, D)
    else:
      print("Encrypted:", C)
      print("Encrypted (decoded):", test.decode(C, digest="hex"))
      print("Key:", K)

  if actions.__contains__("decrypt") and not actions.__contains__("encrypt"):
    C = input("Cipher: ")
    K = input("Key: ")
    D = test.decrypt(str_to_bintensor(C, encoding="hex"), K)
    print("Decrypted:", test.decode(D))
  
  print()

if __name__ == "__main__":
  main()
