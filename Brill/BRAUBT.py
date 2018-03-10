import nltk.corpus, nltk.classify, nltk.tag, itertools
import nltk.tag
from nltk.tag import BrillTaggerTrainer, brill


def backoff_tagger(tagged_sents, tagger_classes, backoff=None):
    if not backoff:
        backoff = tagger_classes[0](tagged_sents)
        del tagger_classes[0]
 
    for cls in tagger_classes:
        tagger = cls(tagged_sents, backoff=backoff)
        backoff = tagger
 
    return backoff


print("Initializing the database")

#brown_review_sents = nltk.corpus.brown.tagged_sents(categories=['reviews'])
#brown_lore_sents = nltk.corpus.brown.tagged_sents(categories=['lore'])
#brown_romance_sents = nltk.corpus.brown.tagged_sents(categories=['romance'])
 
#brown_train = list(itertools.chain(brown_review_sents[:1000], brown_lore_sents[:1000], brown_romance_sents[:1000]))
#brown_test = list(itertools.chain(brown_review_sents[1000:2000], brown_lore_sents[1000:2000], brown_romance_sents[1000:2000]))
 
#conll_sents = nltk.corpus.conll2000.tagged_sents()
#conll_train = list(conll_sents[:4000])
#conll_test = list(conll_sents[4000:8000])
 
treebank_sents = nltk.corpus.treebank.tagged_sents()
treebank_train = list(treebank_sents[:1500])
treebank_test = list(treebank_sents[1500:3000])

word_patterns = [
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
    (r'.*ould$', 'MD'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*ness$', 'NN'),
    (r'.*ment$', 'NN'),
    (r'.*ful$', 'JJ'),
    (r'.*ious$', 'JJ'),
    (r'.*ble$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*ive$', 'JJ'),
    (r'.*ic$', 'JJ'),
    (r'.*est$', 'JJ'),
    (r'^a$', 'PREP'),
]

print("Initializing the train")

raubt_tagger = backoff_tagger(treebank_train, [nltk.tag.AffixTagger,
    nltk.tag.UnigramTagger, nltk.tag.BigramTagger, nltk.tag.TrigramTagger],
    backoff=nltk.tag.RegexpTagger(word_patterns))

templates = brill.fntbl37()
 
trainer = BrillTaggerTrainer(raubt_tagger, templates)
braubt_tagger = trainer.train(treebank_train, max_rules=100, min_score=3)

print("evaluate the model")
print("BRAUBT: ", braubt_tagger.evaluate(treebank_test))