"""Shared, dependency-free safeguards for source-word editing decisions.

These checks are deliberately conservative. Text similarity is retrieval evidence,
not permission to erase a claim, a demonstration, or another speaker's response.
"""
import re

NUMBERS = set(('zero one two three four five six seven eight nine ten eleven '
    'twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty '
    'thirty forty fifty sixty seventy eighty ninety hundred thousand million billion '
    'trillion first second third fourth fifth sixth seventh eighth ninth tenth '
    'eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth '
    'eighteenth nineteenth twentieth thirtieth fortieth fiftieth half quarter').split())
NEGATIONS = set(('no not never neither nor without cannot dont cant wont isnt arent '
    'wasnt werent doesnt didnt havent hasnt hadnt shouldnt wouldnt couldnt mustnt').split())
NUMBER_MODIFIERS={'minus','negative','positive','plus','point'}


def lexemes(text):
    return re.findall(r"[+-]?\d+(?:[.,:/]\d+)*%?|[a-z]+(?:'[a-z]+)?", text.lower().replace('’', "'"))


def normalized(text):
    return re.sub('[^a-z0-9]', '', text.lower())


def facts(text):
    tokens=lexemes(text)
    quantities=[t for t in tokens if any(c.isdigit() for c in t) or t in NUMBERS|NUMBER_MODIFIERS]
    protected={normalized(t) for t in tokens if normalized(t) in NEGATIONS}
    # Decimal points, signs, and order matter: 1.5 is not 5.1, and one hundred
    # fifty is not fifty hundred one. A set of stripped tokens lost this evidence.
    if quantities:protected.add('quantity:'+' '.join(quantities))
    return protected


def quoted_or_demonstrated(text):
    if any(mark in text for mark in ('"', '“', '”', '‘', '’ ')):
        return True
    if re.search(r"(?:^|\s)'[a-zA-Z]",text):return True
    text = ' '.join(lexemes(text))
    return any(phrase in text for phrase in (
        'for example', 'an example', 'the example', 'example of', 'repeat after me',
        'the phrase', 'the word', 'he said', 'she said', 'they said', 'i said',
        'a stutter', 'the stutter', 'demonstrate', 'demonstrating', 'quoted', 'quotation'))


def compatible_words(first, second=()):
    words = list(first) + list(second)
    if not words:
        return False
    # Unknown confidence is not evidence of low confidence; an explicit low score
    # is. A single uncertain boundary word can make a cut unsafe.
    if any(w.get('probability', 1) < .60 for w in words):
        return False
    speakers = {str(w.get('speaker', w.get('speaker_id'))) for w in words
                if w.get('speaker', w.get('speaker_id')) is not None}
    return len(speakers) <= 1


def source_words(transcript):
    """Keep segment speaker labels when flattening adjacent processing chunks."""
    words = []
    for segment in transcript:
        for word in segment.get('words', []):
            if 0 <= word['end'] - word['start'] < 3:
                speaker = word.get('speaker', word.get('speaker_id',
                    segment.get('speaker', segment.get('speaker_id'))))
                words.append(dict(word, speaker=speaker) if speaker is not None else word)
    return sorted(words, key=lambda w: w['start'])


def sentence_context(words, index):
    """Use the enclosing sentence, including context before a processing boundary."""
    left = index
    while left and index-left < 30:
        previous = words[left-1]
        if previous['text'].rstrip().endswith(('.', '!', '?')) or words[left]['start']-previous['end'] > 2:
            break
        left -= 1
    right = index
    while right+1 < len(words) and right-index < 30:
        if words[right]['text'].rstrip().endswith(('.', '!', '?')) or words[right+1]['start']-words[right]['end'] > 2:
            break
        right += 1
    return ''.join(w['text'] for w in words[left:right+1])
