import json
from pydantic import BaseModel, field_validator
import re

class Speaker(BaseModel):
    talare: str
    parti: str

    @field_validator('talare')
    def clean_talare(cls, value):
        # List of titles to remove from 'talare'
        titles = ["Statsrådet", "Utbildningsminister", "Kultur- och idrottsminister",
                  "Socialminister", "Finansminister", "Justitieminister", "Försvarsminister",
                  "Minister", "Kulturministern", "Justitie- och inrikesministern", "Statsministern", 
                  "Infrastrukturministern", "Klimat- och miljöministern", "Näringsministern",
                  "Arbetsmarknadsminister", "Hälso- och sjukvårdsminister", "Bostadsminister", 
                  "Jämställdhetsminister", "Landshövding", "Europaminister", "Digitaliseringsminister",
                  "Skogsminister", "Livsmedelsminister", "Utrikesminister", "Samhällsbyggnadsminister", 
                  "Tullminister", "Socialförsäkringsminister", "Pensionsminister", "Flyktingminister",
                  "Arbetsmarknadsministern"]

        # Loop over titles and remove any occurrence of the title within 'value'
        for title in titles:
            if title in value:
                value = value.replace(title, "")

        # Remove " replik" at the end if present
        value = re.sub(r'\s*replik$', '', value)

        # Remove anything inside parentheses at the end of the name
        value = re.sub(r'\s*\(.*\)$', '', value)

        # Strip any leading or trailing whitespace after removal
        value = value.strip()

        return value

    @field_validator('parti')
    def clean_parti(cls, value):
        # Remove the party affiliation (inside parentheses)
        value = re.sub(r'\s*\(.*\)$', '', value)
        return value

if __name__ =="__main__":
    # Input and Output file paths
    input_file = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json"
    output_file = "/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json"

    # Reading the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Processing each item in the data
    for item in data:
        if 'talare' in item:
            # Clean 'talare' and 'parti' fields using Speaker model
            item['talare'] = Speaker(**item).talare

    # Saving the cleaned data back to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Cleaned data saved to: {output_file}")
