

class PerplexitySubsampler:
    def __init__(self, kenlm_model):
        self.kenlm_model = kenlm_model

    def __call__(self, text):
        return kenlm_model_en.get_perplexity(cleaned_text)

class 