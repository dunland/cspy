import json

with open('settings/cityscopy.json') as f:
    data = json.load(f)
    data['cityscopy']['cam_id'] = 4


print(data)
print(data['cityscopy']['cam_id'])

with open('settings/cityscopy.json', 'w') as output_file:
    json.dump(data, output_file)
