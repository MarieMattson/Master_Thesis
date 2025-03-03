import json
from pydantic import BaseModel, field_validator
import re

class Speaker(BaseModel):
    talare: str

    @field_validator('talare')
    def clean_talare(cls, value):
        # List of titles to remove
        titles = [
            "Statsrådet", "Utbildningsminister", "Kultur- och idrottsminister",
            "Socialminister", "Finansminister", "Justitieminister", "Försvarsminister",
            "Minister", "Talmannen"
        ]
        
        # Remove any title at the start of the name
        pattern = r'^(?:' + '|'.join(titles) + r')\s+'
        value = re.sub(pattern, '', value)

        # Remove " replik" at the end if present
        value = re.sub(r'\s*replik$', '', value)

        # Remove anything inside parentheses at the end of the name
        value = re.sub(r'\s*\(.*\)$', '', value)

        return value

input_file = '/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json' 
output_file = '/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json'  

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if 'talare' in item:
        item['talare'] = Speaker(**item).talare 

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Cleaned data saved to: {output_file}")
