class Closed:
    attrs = ['P', 'PART', 'ADV', 'CONJ', 'CONN', 'PRON', 'fragment', 'REL', 'PRV', 'POST', 'NEG', 'INDEF', 'Q', 'DEM', 'POSS', 'NUM']
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
        self.fragment = {i.split()[0]: i.split()[1] for i in
                       open('../closed_class/Closed_lists_fragment.txt').read().split('\n') if i != ''}