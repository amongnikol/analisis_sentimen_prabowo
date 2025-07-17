from enum import Enum

class SentimentEnum(Enum):
    NEUTRAL = "netral"
    POSITIVE = "positif"
    NEGATIVE = "negatif"

    # karna output dari model int maka dibuatkan mapping
    @classmethod
    def from_index(cls, index: int) -> "SentimentEnum":
        mapping = {
            0: cls.NEUTRAL,
            1: cls.POSITIVE,
            2: cls.NEGATIVE
        }
        return mapping[index]
