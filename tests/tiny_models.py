"""Build tiny randomly initialised local models so real model wrappers can be tested offline.

The weights are random: these helpers test loading, shapes and plumbing, never answer quality.
"""

from __future__ import annotations

from pathlib import Path

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "mill", "employs", "forty", "people"]


def tiny_nli(path: Path) -> Path:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.txt").write_text("\n".join(WORDS))
    BertTokenizerFast(str(path / "vocab.txt")).save_pretrained(path)
    cfg = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        label2id={"contradiction": 0, "entailment": 1, "neutral": 2},
    )
    BertForSequenceClassification(cfg).save_pretrained(path)
    return path


def tiny_causal_lm(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    vocab = {w: i for i, w in enumerate([*WORDS, "<eos>", "i", "don't", "know", "."])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", eos_token="<eos>", pad_token="[PAD]"
    )
    fast.chat_template = (
        "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}the {% endif %}"
    )
    fast.save_pretrained(path)
    GPT2LMHeadModel(
        GPT2Config(vocab_size=len(vocab), n_embd=16, n_layer=1, n_head=2, n_positions=256)
    ).save_pretrained(path)
    return path
