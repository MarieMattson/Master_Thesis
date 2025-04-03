"""
My tinking is that anforande_id is needed, even if it would not be present if using some
trancsriber for this. If it is not there, it is impossible to make comparisons.
    {
        "Unnamed: 0": "370774",
        "dok_hangar_id": "5124227",
        "dok_id": "H90982",
        "dok_titel": "Protokoll 2021/22:82 Onsdagen den 16 mars",
        "dok_rm": "2021/22",
        "dok_nummer": "82",
        "dok_datum": "2022-03-16 00:00:00",
        "avsnittsrubrik": "Vissa säkerhetspolitiska frågor",
        "kammaraktivitet": "ärendedebatt",
        "anforande_id": "3f96d4d6-ccd6-ec11-9170-0090facf175a",
        "anforande_nummer": "51",
        "anforandetext": "<p>Fru talman! Tvåspårspolitiken får inte innebära en bild av att vi har gemensamma intressen här. Vi har hela tiden haft diskussioner om vad vi har för gemensamma intressen med Ryssland när vi pratar energifrågor. De vill sälja, och de vill se till att energi används för att påverka. Vad har vi för gemensamma intressen med dem när vi vill övergå till att ha en annan hållning?</p><p>Man måste vara tydlig med att det här inte är två olika spår. Det är en viktig signalfråga att inte göra mänskliga rättigheter till ett enskilt spår och demokrati och allt annat till ett annat spår. De hör ihop och måste gå hand i hand om de ska få effekt.</p><p>En annan del i diskussionen om kärnvapenfrågorna och kärnvapenförbud är att vi inte uppfattar att en förbudskonvention är en väg framåt. Det är inte en väg framåt. Det är inte en god lösning på något annat än möjligen för dem som vill driva plakatpolitik. Det vi ser är att det inte är en enda av kärnvapenmakterna som deltar - vare sig Storbritannien eller Frankrike, om vi tar de EU-nära, men inte heller de andra stora eller de som inte har erkänt att de har kärnvapen.</p><p>Fru talman! I dag ser vi hur Putin hotar med att använda kärnvapen. Han hotade efter Krim med att han var beredd att använda kärnvapen om han inte fick igenom det han ville där. Nu hotar han igen. Jag har inte hört vare sig Putin, Lavrov eller någon annan säga: Vi har ju en kärnvapenkonvention som förbjuder oss att använda kärnvapen.</p><p>Det är inte en väg framåt. Försök att se att det är NPT som är vägen framåt - om den håller!</p>",
        "intressent_id": "0155487380917",
        "rel_dok_id": "H901UU11",
        "replik": "Y",
        "systemdatum": "2022-05-18 19:06:54",
        "underrubrik": null
    }"""

import json

input_path = '/mnt/c/Users/User/thesis/data_import/filtered_riksdag.json'
output_path = '/mnt/c/Users/User/thesis/data_import/no_metadata_exp1/filtered_riksdag_cleaned.json'

with open(input_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

for entry in data:
    if 'talare' in entry:
        del entry['talare']
    if 'parti' in entry:
        del entry['parti']

with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"Data cleaned and saved to {output_path}")
