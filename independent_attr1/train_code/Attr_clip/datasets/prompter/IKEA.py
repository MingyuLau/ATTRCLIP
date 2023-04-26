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

    __temp_no_gesture = [
        "The person is not making gesture.",
        "The person is not doing anything.",
        "The person is doing nothing.",
    ]

    __temp_other_gesture = [
        "The person is doing something else.",
    ]

    def list_all(self, verb_str: str) -> List[str]:
        if verb_str == "NA":
            return self.__temp_no_gesture
        elif verb_str == "other":
            return self.__temp_other_gesture
        else:
            return [templ.format(verb_str) for templ in self.__temp_interaction_present]

    def random_choice(self, verb_str):
        if verb_str == "NA":
            return random.choice(self.__temp_no_gesture)
        elif verb_str == "other":
            return random.choice(self.__temp_other_gesture)
        else:
            subject_template = random.choice(self.__temp_interaction_present)
            prompt = subject_template.format(verb_str.lower())
            return prompt
