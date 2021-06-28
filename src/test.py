import random
import string
import torch
import os
from numpy import array
from typing import List, Tuple

from config import defaults
from src.util.binary import binlist_to_str, str_to_binlist
from src.anc.model import KeyholderNetwork as ANCKeyholder, AttackerNetwork as ANCAttacker
from src.cryptonet.model import KeyholderNetwork as CNKeyholder, AttackerNetwork as CNAttacker
from src.cryptonet_anc.model import KeyholderNetwork as CNAKeyholder, AttackerNetwork as CNAAttacker


keygen = lambda: ''.join(random.choices(string.ascii_letters + string.digits, k = defaults[defaults["model"]]["blocksize"]))

def evaluate(modelpaths: str = None):
  plain = ''.join(random.choices(string.ascii_letters + string.digits, k = 256))
  key = keygen()

  plaintext = [torch.Tensor(token).unsqueeze(dim=0) for token in str_to_binlist(plain)]
  p = decrypt(encrypt(plain, key, modelpaths), key, modelpaths)

  avg = array([torch.nn.MSELoss()(plaintext[i], p[i]).item() for i in range(len(p))]).sum() / len(p)
  print('Average Reconstruction loss:', "{:.8f}".format(avg))


def evaluate_manual(p: str, d: list):
  txt = [torch.Tensor(token).unsqueeze(dim=0) for token in str_to_binlist(p)]
  avg = array([torch.nn.MSELoss()(d[i], txt[i]).item() for i in range(len(txt))]).sum() / len(txt)
  print('Reconstruction Error:', "{:.8f}".format(avg))


def encrypt(plain: str, key: str, modelpaths: str = None):
  alice, _, _ = load_models(modelpaths)

  plain = str_to_binlist(plain)
  key = next(str_to_binlist(key))
  
  ciphertext = list()
  for token in plain:
    token = torch.Tensor(token).unsqueeze(dim=0)
    K = torch.Tensor(key).unsqueeze(dim=0)
    cipher = alice(torch.cat([token, K], dim=1))
    ciphertext.append(cipher)

  return ciphertext


def decrypt(cipher: List[torch.Tensor], key: str, modelpaths: str = None):
  _, bob, _ = load_models(modelpaths)

  key = next(str_to_binlist(key))
  
  plaintext = list()
  for token in cipher:
    K = torch.Tensor(key).unsqueeze(dim=0)
    plain = bob(torch.cat([token, K], dim=1))
    plaintext.append(plain)

  return plaintext


def decode(plaintext, **kwargs):
  plaintext = [ token.reshape(token.shape[1]).tolist() for token in plaintext ]
  return binlist_to_str(plaintext, **kwargs)


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
  elif defaults["model"] == "cryptonet_anc":
    alice = CNAKeyholder(blocksize=config["blocksize"], name="Alice")
    bob = CNAKeyholder(blocksize=config["blocksize"], name="Bob")
    eve = CNAAttacker(blocksize=config["blocksize"], name="Eve")

  alice.load_state_dict(torch.load(alicepath))
  bob.load_state_dict(torch.load(bobpath))
  eve.load_state_dict(torch.load(evepath))

  if set_eval:
    alice.eval()
    bob.eval()
    eve.eval()

  return alice, bob, eve
