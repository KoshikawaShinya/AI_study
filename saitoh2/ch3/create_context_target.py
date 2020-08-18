import numpy as np


def create_context_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    context = []

    for index in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[index + t])
        contexts.append(cs)

    return np.array(context), np.array(target) 