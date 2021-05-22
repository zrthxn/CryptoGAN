def str_to_binlist(text: str, binlen: int = 2):
  byte = list()

  for b in bytearray(text, "utf8"):
    byte.append(pad(bin(b)[2:]))

  for i in range(0, len(byte), binlen):
    binlist = list()
    string = "".join([ byte[i + j] for j in range(binlen) ])
    for k, s in enumerate(string):
      binlist.append(float(s))
      
    yield binlist


def pad(s: str):
  for _ in range(len(s), 8, 1):
    s = '0' + s
  return s
