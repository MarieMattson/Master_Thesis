import json
from pydantic import BaseModel, field_validator
import re
from typing import Optional

class Speaker(BaseModel):
    talare: Optional[str] = None
    parti: Optional[str] = None

    @field_validator('talare')
    def clean_talare(cls, value):
        if not value or value.strip() == "":
            return "Unknown"

        titles = [
            "Statsrådet", "Utbildningsminister", "Kultur- och idrottsminister",
            "Socialminister", "Finansminister", "Justitieminister", "Försvarsminister",
            "Minister", "Kulturministern", "Justitie- och inrikesministern", "Statsministern", 
            "Infrastrukturministern", "Klimat- och miljöministern", "Näringsministern",
            "Arbetsmarknadsminister", "Hälso- och sjukvårdsminister", "Bostadsminister", 
            "Jämställdhetsminister", "Landshövding", "Europaminister", "Digitaliseringsminister",
            "Skogsminister", "Livsmedelsminister", "Utrikesminister", "Samhällsbyggnadsminister", 
            "Tullminister", "Socialförsäkringsminister", "Pensionsminister", "Flyktingminister",
            "Arbetsmarknadsministern"
        ]

        for title in titles:
            value = value.replace(title, "")

        value = re.sub(r'\s*replik$', '', value)
        value = re.sub(r'\s*\(.*\)$', '', value)
        return value.strip() or "Unknown"

    @field_validator('parti')
    def clean_parti(cls, value):
        if not value or value.strip() == "":
            return "Unknown"
        return re.sub(r'\s*\(.*\)$', '', value).strip() or "Unknown"
    
if __name__ == "__main__":
    input_file = "/mnt/c/Users/User/thesis/data_import/data_large_size/unfiltered_riksdag.json"
    output_file = "/mnt/c/Users/User/thesis/data_import/data_large_size/filtered_riksdag.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        speaker = Speaker(**item)
        item['talare'] = speaker.talare
        item['parti'] = speaker.parti

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Cleaned data saved to: {output_file}")
