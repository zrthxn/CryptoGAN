from typing import List, Tuple
import torch
import os
from numpy import array

from config import defaults
from src.util.binary import str_to_binlist
from src.anc.model import KeyholderNetwork as ANCKeyholder, AttackerNetwork as ANCAttacker
from src.anc.datagen import KeyGenerator as ANCKeyGen, PlainGenerator as ANCPlainGen
from src.cryptonet.model import KeyholderNetwork as CNKeyholder, AttackerNetwork as CNAttacker
from src.cryptonet.datagen import KeyGenerator as CNKeyGen, PlainGenerator as CNPlainGen


def evaluate(modelpaths: str = None):
  key = "KEYDKEYDKEYDKEYD"
  plain = "Hello World!1234"

  plaintext = [torch.Tensor(token).unsqueeze(dim=0) for token in str_to_binlist(plain)]
  p = decrypt(encrypt(plain, key), key)

  avg = array([torch.nn.MSELoss()(plaintext[i], p[i]).item() for i in range(len(p))]).sum()/len(p)
  print(avg)


def encrypt(plain: str, key: str, modelpaths: str = None):
  alice, _, _ = load_models(modelpaths)

  plain = str_to_binlist(plain)
  key = str_to_binlist(key)
  key = torch.Tensor(next(key)).unsqueeze(dim=0)
  
  ciphertext = list()
  for token in plain:
    token = torch.Tensor(token).unsqueeze(dim=0)
    cipher = alice(torch.cat([token, key], dim=1))
    ciphertext.append(cipher)

  return ciphertext


def decrypt(cipher: List[torch.Tensor], key: str, modelpaths: str = None):
  _, bob, _ = load_models(modelpaths)

  key = str_to_binlist(key)
  key = torch.Tensor(next(key)).unsqueeze(dim=0)
  
  plaintext = list()
  for token in cipher:
    plain = bob(torch.cat([token, key], dim=1))
    plaintext.append(plain)

  return plaintext


def load_models(modelpaths: str = None, set_eval: bool = True) -> Tuple[torch.nn.Module]:
  if modelpaths is not None:
    alicepath, bobpath, evepath = modelpaths
  else:
    files = [f for f in os.listdir(f'models/{defaults["model"]}') if f.endswith('.mdl')]
    for f in files:
      if f.startswith('Alice'):
        alicepath = os.path.join(f'models/{defaults["model"]}', f)
      elif f.startswith('Bob'):
        bobpath = os.path.join(f'models/{defaults["model"]}', f)
      elif f.startswith('Eve'):
        evepath = os.path.join(f'models/{defaults["model"]}', f)
      else:
        raise ImportError("Unknown model " + f)
    pass

  config = defaults[defaults["model"]]
  if defaults["model"] == "anc":
    alice = ANCKeyholder(blocksize=config["blocksize"], name="Alice")
    bob = ANCKeyholder(blocksize=config["blocksize"], name="Bob")
    eve = ANCAttacker(blocksize=config["blocksize"], name="Eve")
  elif defaults["model"] == "cryptonet":
    alice = CNKeyholder(blocksize=config["blocksize"], name="Alice")
    bob = CNKeyholder(blocksize=config["blocksize"], name="Bob")
    eve = CNAttacker(blocksize=config["blocksize"], name="Eve")

  alice.load_state_dict(torch.load(alicepath))
  bob.load_state_dict(torch.load(bobpath))
  eve.load_state_dict(torch.load(evepath))

  if set_eval:
    alice.eval()
    bob.eval()
    eve.eval()

  return alice, bob, eve
