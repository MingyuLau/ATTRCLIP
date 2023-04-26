import pdb
import random
from typing import List


class Prompter(object):
    __temp_interaction_present = [
        # present_participle: 现在分词
        "A person is {}.",
        "Someone is {}.",
        "Somebody is {}.",
        "A person is {} with hands.",
        "Someone is {} with hands.",
        "Somebody is {} with hands.",
        # infinitive:         动词不定式
        "A person is using hands to {}.",
        "Someone is using hands to {}.",
        "Somebody is using hands to {}.",
    ]

    __temp_no_gesture = [
        "The person is not making gesture.",
        "The person is not doing anything.",
        "The person is doing nothing.",
    ]

    def list_all(self, verb_str: str) -> List[str]:
        if verb_str == "SIL":
            return self.__temp_no_gesture
        else:
            return [templ.format(verb_str) for templ in self.__temp_interaction_present]

    def random_choice(self, verb_str):
        if verb_str == "SIL":
            return random.choice(self.__temp_no_gesture)
        else:
            subject_template = random.choice(self.__temp_interaction_present)
            prompt = subject_template.format(verb_str.lower())
            return prompt
# 在自然语言处理中，SIL经常表示为silence
if __name__ == "__main__":
    prompter = Prompter()
    a = prompter.random_choice('SIL')
    pdb.set_trace()
    print('end')