class Subsampler:
    def __init__(self, func):
        self.func = func

    def __call__(self, text):
        error, value = None, None
        try:
            if isinstance(self.func, list):
                value = [func(text) for func in self.func]
            else:
                value = self.func(text)
            value = str(value).replace("'", '"').encode("utf-8")
        except Exception as err:
            error = str(err)
        return value, error
