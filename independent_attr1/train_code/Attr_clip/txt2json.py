import json


with open('/home/user/lmy/DATA/independent_attr1/Attr_Sample/semantic_attr_train_VG.json', 'r') as f:
    attributes = json.load(f)
attribute_dict = {}
for i, attribute in enumerate(attributes['pair']):
    attribute_dict[str(tuple(attribute[1]))] = i

with open('attr2id_vg.json', 'w') as f:
    json.dump(attribute_dict, f, indent=4)



# attribute -> prompt 
# accuracy image gt: attribute
#           image -> image feature  
#           attr集合 413个  Prompter train val test