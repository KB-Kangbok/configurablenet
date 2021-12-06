class Wrapper:
    def __init__(self, ray) -> None:
        self.Ray = ray
        self.DL = None
        self.optim = None
        self.config = None

    def set_space(self, config, dl, optim):
        self.config = config
        self.dl = dl
        self.optim = optim
        return

    def data_loader(self):
        return

    def train(self):
        return