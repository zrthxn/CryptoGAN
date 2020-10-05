from datetime import date, time

def modelname(name, *args):
  return name + '-'.join(args) + '_' + str(date.today())

if __name__ == "__main__":
  print(modelname('ModelName', 'Size', 'Version'))
  