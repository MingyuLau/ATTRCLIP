import random
from typing import List, Optional

class Prompter(object):
    __temp_interaction = [
    # present_participle:
        "I am {}.",
        "I am {} with my hands.",
        "My hands are {}.",
        # "My left hand is {}.",
        # "My right hand is {}.",
        # "My both hands are {}.",
        # "I am {} with my left hand.",
        # "I am {} with my right hand.",
        # "I am {} with my both hands.",

    # infinitive:
        "I am using my hands to {}.",
        # - "I am using my left hand to {}.",
        # - "I am using my right hand to {}.",
        # - "I am using my both hands to {}.",
    ]
    
    __temp_object = [
        "something",
        "some objects",
        "an object",
    ]

    def list_all(self, verb_str: str, noun_str: Optional[str]=None) -> List[str]:
        return [verb_str]

    def random_choice(self, verb_str: str, noun_str: Optional[str]=None) -> str:
        assert noun_str is None
        return verb_str
