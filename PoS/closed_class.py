class Closed:
    attrs = ['adverb', 'compliment', 'conjunct', 'connector', 'preverb', 'pronoun', 'q_word']
    def __init__(self):
        self.adverb = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_adverbs.txt').read().split('\n') if i != ''}
        self.compliment = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_complementizers.txt').read().split('\n') if i != ''}
        self.conjunct = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_conjunctions.txt').read().split('\n') if i != ''}
        self.connector = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_connectors.txt').read().split('\n') if i != ''}
        self.preverb = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_preverbs.txt').read().split('\n') if i != ''}
        self.pronoun = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_pronouns.txt').read().split('\n') if i != ''}
        self.q_word = {i.split()[0]: i.split()[1] for i in
                           open('../closed_class/Closed_lists_q-words.txt').read().split('\n') if i != ''}
