from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random
stop_words = list(set(stopwords.words('english')))


def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


def synonym_replacement(words, n):
    words = words.split()

    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence


def random_deletion(words, p):
    words = words.split()

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words[0]

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return words[rand_int]

    sentence = ' '.join(new_words)

    return sentence


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1

        if counter > 3:
            return new_words

    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_swap(words, n):
    words = words.split()
    new_words = words.copy()

    for _ in range(n):
        new_words = swap_word(new_words)

    sentence = ' '.join(new_words)

    return sentence


def random_insertion(words, n):
    words = words.split()
    new_words = words.copy()

    for _ in range(n):
        add_word(new_words)

    sentence = ' '.join(new_words)
    return sentence


def add_word(new_words):
    synonyms = []
    counter = 0

    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return

    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def random_augmentation(words, n=1):
    for _ in range(20):
        random_percent = random.random()
        if random_percent <= 0.7:
            new_words = synonym_replacement(words, n)
        elif random_percent <= 0.8:
            new_words = random_deletion(words, n)
        elif random_percent <= 0.9:
            new_words = random_swap(words, n)
        elif random_percent <= 1:
            new_words = random_insertion(words, n)
        if new_words != words:
            return new_words
    return new_words + ' ' + stop_words[random.randint(0, 178)]


if __name__ == '__main__':
    words = 'journal of risk management'
    print(random_augmentation(words))
