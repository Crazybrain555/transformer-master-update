"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import spacy


class Tokenizer:

    def __init__(self):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(self, text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]



if __name__ == '__main__':
    tokenizer_fortest = Tokenizer()

    # 示例的复杂德语句子
    print(tokenizer_fortest.tokenize_de(
        "Hans's Fahrrad, das er aus dem Fahrradladen geholt hat, ist brandneu und glänzt in der Sonne. Obwohl es noch nicht ganz zusammengesetzt ist, kann er es kaum erwarten, seine erste Fahrt zu machen!"))
    # 预期输出: ['Hans', "'s", 'Fahrrad', ',', 'das', 'er', 'aus', 'dem', 'Fahrradladen', 'geholt', 'hat', ',', 'ist',
    # 'brandneu', 'und', 'glänzt', 'in', 'der', 'Sonne', '.', 'Obwohl', 'es', 'noch', 'nicht', 'ganz', 'zusammengesetzt',
    # 'ist', ',', 'kann', 'er', 'es', 'kaum', 'erwarten', ',', 'seine', 'erste', 'Fahrt', 'zu', 'machen', '!']

    # 示例的复杂英语句子
    print(tokenizer_fortest.tokenize_en(
        "John's new bike, which he'd just picked up from the bike store, was brand new and gleaming in the sunlight. Even though it wasn't fully assembled yet, he couldn't wait to take it for its first ride!"))
    # 预期输出: ['John', "'s", 'new', 'bike', ',', 'which', 'he', "'d", 'just', 'picked', 'up', 'from', 'the', 'bike',
    # 'store', ',', 'was', 'brand', 'new', 'and', 'gleaming', 'in', 'the', 'sunlight', '.', 'Even', 'though', 'it',
    # 'was', "n't", 'fully', 'assembled', 'yet', ',', 'he', 'could', "n't", 'wait', 'to', 'take', 'it', 'for', 'its',
    # 'first', 'ride', '!']

    pass



