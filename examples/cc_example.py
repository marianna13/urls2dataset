from urls2dataset import urls2dataset
import pandas as pd
import time
import kenlm
import sentencepiece
from huggingface_hub import cached_download, hf_hub_url
import re
from typing import Dict
from requests.exceptions import HTTPError
from riverbed.kenlm_manager import *
import json
from pii_transform.api.e2e.multilang import MultiPiiTextProcessor
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

KENLM_MODEL_REPO = "siddhesh1793/kenlm"

DATAPATH = Path('./')
configfile = DATAPATH / 'piisa-config.yml'

proc = MultiPiiTextProcessor(lang=["en", "es"], config=configfile, 
                             keep_piic=False, debug=None)



def replace_personal_data(text):
    try:
        pii_cleaned_text = proc(' '.join(text), lang="en")
        
    except:
        pii_cleaned_text = ' '.join(text)
    return pii_cleaned_text

class SentencePiece:
    def __init__(
        self,
        model: str,
    ):
        super().__init__()
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(model))

    def do(self, text: dict) -> dict:
        tokenized = self.sp.encode_as_pieces(text)
        return " ".join(tokenized)


class KenlmModel:
    digit_re: re.Pattern = re.compile(r"\d")
    unicode_punct: Dict[str, str] = {
        ",": ",",
        "?": ".",
        "?": ",",
        "„": '"',
        "”": '"',
        "“": '"',
        "«": '"',
        "»": '"',
        "1": '"',
        "?": '"',
        "?": '"',
        "«": '"',
        "»": '"',
        "´": "'",
        ":": ":",
        ":": ":",
        "?": "?",
        "!": "!",
        "(": "(",
        ")": ")",
        ";": ";",
        "–": "-",
        "—": " - ",
        ".": ". ",
        "~": "~",
        "’": "'",
        "…": "...",
        "?": "-",
        "<": "<",
        ">": ">",
        "?": "[",
        "?": "]",
        "%": "%",
        "?": "-",
    }
    unicode_punct_re = re.compile(f"[{''.join(unicode_punct.keys())}]")
    non_printing_chars_re = re.compile(
        f"[{''.join(map(chr, list(range(0,32)) + list(range(127,160))))}]"
    )
    kenlm_model_dir = None
    sentence_piece_model_dir = None

    def __init__(
        self,
        model_dataset: str,
        language: str,
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ):
        self.download_kenlm_model(model_dataset, language)
        try:
            self.model = kenlm.Model(self.kenlm_model_dir)
            self.tokenizer = SentencePiece(self.sentence_piece_model_dir)
        except OSError:
            os.remove(self.kenlm_model_dir)
            if os.path.exists(self.sentence_piece_model_dir):
                os.remove(self.sentence_piece_model_dir)
            raise OSError(
                "File was corrupt and should have been removed. Please, retry."
            )
        self.accent = remove_accents
        self.case = lower_case
        self.numbers = normalize_numbers
        self.punct = punctuation

    @classmethod
    def from_pretrained(
        cls,
        model_dataset: str,
        language: str,
        lower_case: bool = False,
        remove_accents: bool = False,
        normalize_numbers: bool = True,
        punctuation: int = 1,
    ):
        return cls(
            model_dataset,
            language,
            lower_case,
            remove_accents,
            normalize_numbers,
            punctuation,
        )

    def pp(self, log_score, length):
        return 10.0 ** (-log_score / length)

    def get_perplexity(self, doc: str, normalize_cc_net: bool = True):
        if normalize_cc_net:
            doc = self.normalize(
                doc,
                accent=self.accent,
                case=self.case,
                numbers=self.numbers,
                punct=self.punct,
            )
        # Tokenize (after normalizing): See https://github.com/facebookresearch/cc_net/blob/bda555bd1cf1ee2e0b925363e62a61cd46c8b60d/cc_net/mine.py#L352 for full pipeline
        doc = self.tokenizer.do(doc)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = self.model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return round(self.pp(doc_log_score, doc_length), 1)

    def normalize(
        self,
        line: str,
        accent: bool = True,
        case: bool = True,
        numbers: bool = True,
        punct: int = 1,
    ) -> str:
        line = line.strip()
        if not line:
            return line
        if case:
            line = line.lower()
        if accent:
            line = self.strip_accents(line)
        if numbers:
            line = self.digit_re.sub("0", line)
        if punct == 1:
            line = self.replace_unicode_punct(line)
        elif punct == 2:
            line = self.remove_unicode_punct(line)
        line = self.remove_non_printing_char(line)
        return line

    def strip_accents(self, line: str) -> str:
        """Strips accents from a piece of text."""
        nfd = unicodedata.normalize("NFD", line)
        output = [c for c in nfd if unicodedata.category(c) != "Mn"]
        if len(output) == line:
            return line
        return "".join(output)

    def replace_unicode_punct(self, text: str) -> str:
        return "".join(self.unicode_punct.get(c, c) for c in text)

    def remove_unicode_punct(self, text: str) -> str:
        """More aggressive version of replace_unicode_punct but also faster."""
        return self.unicode_punct_re.sub("", text)

    def remove_non_printing_char(self, text: str) -> str:
        return self.non_printing_chars_re.sub("", text)

    def download_kenlm_model(self, model_dataset: str, language: str):
        try:
            kenlm_model_url = hf_hub_url(
                KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.arpa.trie.bin"
            )
            self.kenlm_model_dir = cached_download(kenlm_model_url)
        except HTTPError:
            kenlm_model_url = hf_hub_url(
                KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.arpa.bin"
            )
            self.kenlm_model_dir = cached_download(kenlm_model_url)
        sentence_piece_model_url = hf_hub_url(
            KENLM_MODEL_REPO, filename=f"{model_dataset}/{language}.sp.model"
        )
        self.sentence_piece_model_dir = cached_download(sentence_piece_model_url)


kenlm_model_books = KenlmModel.from_pretrained('the_pile_books3', 'en')

kenlm_model_en = load_kenlm_model("en", pretrained_models=["ccnet/wikipedia"])
kenlm_model_en = kenlm_model_en['ccnet/wikipedia']

def get_extension(url: str) -> str:
    """Parse the URL using the urlparse method
    Get the file name and extension from the parsed URL
    Return the file extension"""
    parsed_url = urlparse(url)
    try:
        filename, file_ext = parsed_url.path.rsplit(".", maxsplit=1)
    except ValueError:
        file_ext = ""
    return file_ext



def text2chunks(string):
    # Create an empty list to store the resulting substrings.
    result = []

    # Keep track of the start and end indices of the previous match.
    prev_end = 0

    # Iterate over the input string, looking for instances of the ###<text>#<digits> pattern.
    for match in re.finditer(r"###[^#]+#\d+###", string):
        # Get the start and end indices of the current match.
        start = match.start()
        end = match.end()

        # Extract the text between the previous match and the current match.
        substring = string[prev_end:start]

        # Add the substring and the current match to the list of substrings.
        result.append(substring)
        result.append(match.group(0))

        # Update the previous end index.
        prev_end = end

    # Extract the text after the last match.
    substring = string[prev_end:]

    # Add the final substring to the list of substrings.
    result.append(substring)

    result = list(filter(lambda x: x != "", result))
    result = list(filter(lambda x: x != " ", result))

    return result

def get_perplexities(text):
    cleaned_text = replace_personal_data(text)
    books_p = kenlm_model_books.get_perplexity(cleaned_text)
    en_p = kenlm_model_en.get_perplexity(cleaned_text)
    return [books_p, en_p]

def filter_pereplexities(perp):
    perplexity_score_books, perplexity_score_en = json.loads(perp)
    return (perplexity_score_books > 1200 or perplexity_score_en > 70_000)

if __name__ == "__main__":


    urls2dataset(
        url_list="s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00009-ip-10-60-113-184.ec2.internal.warc.gz",
        input_format="cc",
        output_format="parquet",
        output_folder="data",
        processes_count=16,
        number_sample_per_shard=1000,
        thread_count=16,
        config={
            'media_elems':True,
            'save_media_struct': True
            },
        postprocess_func=get_perplexities,
        filters_config={
            'filter_func':filter_pereplexities,
            'filter_col':'postproc_value'
        }
    )

    print(pd.read_parquet("data/00000.parquet"))
