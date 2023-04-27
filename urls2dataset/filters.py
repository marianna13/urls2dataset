class Filter:
    def __init__(self, filter_func, filter_col):
        self.filter_func = filter_func
        self.filter_col = filter_col

    def __call__(self, sample):
        return self.filter_func(sample[self.filter_col])
