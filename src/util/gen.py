from tqdm import tqdm
import multiprocessing

class Generator():
  _cpus = multiprocessing.cpu_count()
  _list = dict()

  def __init__(self, blocksize: int, batchlen: int) -> None:
    self.blocksize = blocksize
    self.batchlen = batchlen

  def gen(self):
    """Generate a single instance
    """
    raise NotImplementedError

  def next(self, key: int = None):
    """Generate or fetch one batch from generated data.

    Args:
      key (int, optional): Key of the batch to fetch. Defaults to None.

    Returns:
      List: Batch. New if `key` doesn't exist. Not saved if `key` is `None`
    """
    if key is None:
      return [ self.gen() for _ in range(self.batchlen)]

    if key not in self._list.keys():
      gen = [ self.gen() for _ in range(self.batchlen)]
      self._list.update({ str(key): gen }) 

    return self._list[str(key)]

  def batchgen(self, batches: int):
    ite = tqdm(range(batches)) if not self.silent else range(batches)
    return [ self.next(i) for i in ite ]

  def spawn(self):
    # multiprocessing.
    pass
  