"""Pure transcript fixtures; importing these never runs a model or a test."""


def fixture(texts, starts=None):
    transcript, ranges = [], []
    starts = starts or [1+i*12 for i in range(len(texts))]
    for text, start in zip(texts, starts):
        words = [dict(text=' '+t, start=start+i*.2, end=start+i*.2+.16, probability=.95)
                 for i,t in enumerate(text.split())]
        transcript.append(dict(words=words))
        ranges.append([start-.075, words[-1]['end']+.12])
    return transcript, ranges
