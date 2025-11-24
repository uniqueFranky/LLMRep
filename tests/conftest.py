from pytest import fixture


@fixture(scope='session')
def tokens_without_rep():
    return ['I', 'like', 'apple']

@fixture(scope='session')
def tokens_with_3gram_rep():
    return ['I', 'like', 'apple', 'I', 'like', 'apple', 'I', 'like', 'apple']

