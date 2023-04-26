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

    def list_all(self, verb_str: str, noun_str: Optional[str] = None) -> List[str]:
        if noun_str is None:
            obj_template_list = self.__temp_object
        else:
            obj_template_list = [noun_str]

        prompts_per_class = []
        for object_template in obj_template_list:
            for subject_template in self.__temp_interaction:
                hoi_text = verb_str.replace(
                    "-", " ") + " " + object_template  # e.g. hold something
                prompts_per_class.append(
                    subject_template.format(hoi_text))
        return prompts_per_class

    def random_choice(self, verb_str: str, noun_str: Optional[str] = None) -> str:
        if noun_str is None:
            noun_str = random.choice(self.__temp_object)

        hoi_text = verb_str.lower().replace("-", " ") + " " + \
            noun_str  # e.g. hold something

        subject_template = random.choice(self.__temp_interaction)
        prompt = subject_template.format(hoi_text)
        return prompt
