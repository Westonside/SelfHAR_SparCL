class EarlyStop():
    def __init__(self, patience: int, testing=False):
       self.count = 0
       self.patience = patience
       self.best_loss = float("inf")
       self.testing_counter = 0
       self.testing = testing

    def check(self, valid_loss: int) -> bool:
        if self.testing:
            self.testing_counter +=1
            if self.testing_counter >= 3:
                return True
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
        self.testing_counter = 0