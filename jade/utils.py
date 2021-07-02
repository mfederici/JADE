from tqdm import notebook, tqdm


class TimeInterval:
    def __init__(self, freq):
        self.freq = int(freq.split(' ')[0])
        self.unit = freq.split()[1].lower()
        self.last_logged = 0

    def is_time(self, model):
        if not hasattr(model, self.unit):
            raise Exception('Invalid unit %s' % (self.unit))
        curr_value = getattr(model, self.unit)

        return (curr_value-self.last_logged) >= self.freq

    def update(self, model):
        self.last_logged = getattr(model, self.unit)

    def percentage(self, model):
        curr_value = getattr(model, self.unit)
        return int((curr_value-self.last_logged)/self.freq * 100)


def tqdm_wrap(*args, **kwargs):
    if isnotebook():
        return notebook.tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter