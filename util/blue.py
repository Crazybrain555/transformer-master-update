import math
from collections import Counter
import numpy as np


def generate_ngrams(seq, n):
    """
    Generate n-grams for a given sequence
    """
    return [tuple(seq[i:i + n]) for i in range(len(seq) + 1 - n)]


def bleu_stats(hypothesis, reference):
    """
    Compute statistics required for BLEU score calculation
    """
    stats = [len(hypothesis), len(reference)]

    for n in range(1, 5):  # 1-gram to 4-gram
        hyp_ngrams = Counter(generate_ngrams(hypothesis, n))
        ref_ngrams = Counter(generate_ngrams(reference, n))

        # Count of n-grams in both hypothesis and reference
        overlap_ngrams = hyp_ngrams & ref_ngrams
        stats.append(sum(overlap_ngrams.values()))

        # Total n-grams in hypothesis
        stats.append(len(generate_ngrams(hypothesis, n)))

    return stats


def bleu(stats):
    """
    Compute BLEU score
    """
    if 0 in stats:
        return 0

    c, r = stats[:2]
    precision = sum(math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])) / 4

    return math.exp(min([0, 1 - float(r) / c]) + precision)


def get_bleu(hypotheses, references):
    """
    Calculate BLEU score for a set of hypotheses and references
    """
    # Initialize to zeros
    stats = np.zeros(10)

    for hyp, ref in zip(hypotheses, references):
        stats += bleu_stats(hyp, ref)

    return 100 * bleu(stats)


def idx_to_word(x, vocab):
    """
    Convert sequence of word indices to words
    """
    words = [vocab.get_itos()[i] for i in x if '<' not in vocab.get_itos()[i]]
    return " ".join(words)
