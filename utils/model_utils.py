class EarlyStop():
    def __init__(self, patience: int):
       self.count = 0
       self.patience = patience
       self.best_loss = float("inf")

    def check(self, valid_loss: int) -> bool:
        if self.best_loss > valid_loss:
            self.best_loss = valid_loss
            self.count = 0
        else:
            self.count += 1
        if self.count > self.patience:
            return True
        else:
            return False

    def reset(self):
        self.count = 0
        self.best_loss = float("inf")