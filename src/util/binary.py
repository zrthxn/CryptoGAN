def str_to_binlist(text: str, binlen: int = 2, values: tuple = (-1, 1)):
  byte = list()

  for b in bytearray(text, "utf8"):
    byte.append(pad(bin(b)[2:]))

  for i in range(0, len(byte), binlen):
    binlist = list()
    string = "".join([ byte[i + j] for j in range(binlen) ])
    for _, s in enumerate(string):
      binlist.append(float(values[0] if s == "0" else values[1]))
      
    yield binlist


def binlist_to_str(binlist: list, decision_point: int = 0):
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
    string[i] = _int.to_bytes((_int.bit_length() + 7) // 8, "big").decode()

  return "".join(string)


def pad(s: str):
  for _ in range(len(s), 8, 1):
    s = '0' + s
  return s
