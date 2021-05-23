from torch.tensor import Tensor
from config import defaults

def str_to_binlist(text: str, values: tuple = (-1, 1), encoding: str = "utf-8"):
  binlen = defaults[defaults["model"]]["blocksize"] // 8
  byte = list()
  
  if len(text) % binlen != 0:
    text += "~"

  if encoding == "utf-8":
    byte = [ pad(bin(b).lstrip("0b")) for b in bytearray(text, "utf8") ]
  elif encoding == "hex":
    byte = [ pad(bin(b).lstrip("0b")) for b in bytearray.fromhex(text) ]

  for i in range(0, len(byte), binlen):
    binlist = list()
    string = "".join([ byte[i + j] for j in range(binlen) ])
    for _, s in enumerate(string):
      binlist.append(float(values[0] if s == "0" else values[1]))
      
    yield binlist


def str_to_bintensor(text: str, **kwargs):
  return [ Tensor(b).unsqueeze(dim=0) for b in str_to_binlist(text, **kwargs) ]


def binlist_to_str(binlist: list, digest: str = "ascii", decision_point: int = 0):
  string = list()
  
  for byte in binlist:
    if len(byte) % 8 != 0:
      raise IndexError("Binary list must be divisible into bytes.")

  for token in binlist:
    token = [token[i:i + 8] for i in range(0, len(token), 8)]
    for byte in token:
      string.append("".join([ 
        ("0" if bit < decision_point else "1") 
      for bit in byte]))

  for i, char in enumerate(string):
    _int = int(char, 2)
    if digest == "ascii":
      try:
        string[i] = _int.to_bytes((_int.bit_length() + 7) // 8, "big").decode()
      except:
        string[i] = hex(_int).lstrip("0x").rstrip("L")
    elif digest == "hex":
      string[i] = hex(_int).lstrip("0x").rstrip("L")

  return "".join(string).replace("~", " ")


def pad(s: str):
  for _ in range(len(s), 8, 1):
    s = '0' + s
  return s
