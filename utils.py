
class dotdict(dict):

    def __init__(self, dictionary=None):
        if dictionary is not None:
            super().__init__(dictionary)
            for key in dictionary:
                if type(dictionary[key]) == dict:
                    self[key] = dotdict(dictionary[key])
        else:
            super().__init__()

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
