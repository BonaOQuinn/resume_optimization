"""An in-memory WordPiece tokenizer keeps splitting tests offline."""

from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def make_tokenizer():
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
             "the", "team", "built", "an", "api", "for", "users",
             "python", "engineer", "project", "course", "skill", ".",
             ":", "-", "first", "last", "sentence", "title"]
    vocab = {word: index for index, word in enumerate(words)}
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", model_max_length=256,
    )
