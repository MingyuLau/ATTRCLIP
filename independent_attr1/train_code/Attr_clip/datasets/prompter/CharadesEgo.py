import random
from typing import List


class Prompter(object):
    __temp_interaction_present = [
        # present_participle:
        "A person is {}.",
        "Someone is {}.",
        "Somebody is {}.",
        "A person is {} with hands.",
        "Someone is {} with hands.",
        "Somebody is {} with hands.",
        # infinitive:
        "A person is using hands to {}.",
        "Someone is using hands to {}.",
        "Somebody is using hands to {}.",
    ]

    def list_all(self, verb_str: str) -> List[str]:
        return [templ.format(verb_str) for templ in self.__temp_interaction_present]

    def random_choice(self, verb_str):   # verb_str: ["do a","do b","do c"]
        # str = ", ".join(verb_str)
        subject_template = random.choice(self.__temp_interaction_present)
        prompt = subject_template.format(verb_str.lower())
        return prompt
