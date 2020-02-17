from enum import Enum, unique


@unique
class WordType(Enum):
    word = 'word'
    url = 'url'
    email = 'email'
    emoji = 'emoji'
    acronymn = 'acronymn'
    mix = 'mix'
