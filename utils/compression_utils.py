from conf import settings

class PR_scheduler():
    def __init__(self, modification_num: int):
        self.modification_num = modification_num
        self.tolerance_ct = settings.TOLERANCE_CT
    
    def reset(self, modification_num: int):
        self.modification_num = modification_num
        self.tolerance_ct = settings.TOLERANCE_CT

    def step(self, model_index: int):
        if model_index == 0:
            # means original net is better
            self.tolerance_ct -= 1
        else:
            # means generated net is better, reset counter
            self.tolerance_ct = settings.TOLERANCE_CT
        if self.tolerance_ct <= 0:
            self.modification_num //= 2
            self.tolerance_ct = settings.TOLERANCE_CT
        if self.modification_num < settings.MODIFICATION_MIN_NUM:
            return None
        else:
            return self.modification_num
