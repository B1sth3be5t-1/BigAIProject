import pandas as ps
import numpy as np
import pycountry
import pgeocode

data = ps.read_csv("CSV_Files/fall-2021-noaddr.csv")

def getDistances(data):
    # read in some column data for zips and country codes
    zip_col = data['ZIP']
    international_col = data['INTERNATIONAL']
    country_col = data['COUNTRY']

    # Some items in COUNTRY use two-character indicators, while
    # some others use a full country name. We'll run the country name
    # ones through
    print(set(country_col))

    # Remove trailing -00000 and region-specific codes
    stripped_zips = map(lambda s: str(s)[:min(5, len(str(s)))], zip_col)

    country_col = country_col.map(lambda c: str(c))

    count = 0
    new_countries = []
    for c in country_col:
        if c == "nan":
            country = pycountry.countries.get(alpha_2="us")
        elif len(c) == 2:
            country = pycountry.countries.get(alpha_2=c)
        else:
            # so turkey gives us problems. We have to change it to the official name
            if c == "Turkey":
                c = "turkiye"
            country = pycountry.countries.search_fuzzy(c)[0]
        count += 1
        new_countries.append(country.alpha_2)

    # for s in set(new_countries):
    #    print(s)

    coords = []
    nominatims = {}
    num_bad_bois = 0

    for country_code, postal_code in zip(new_countries, stripped_zips):
        try:
            if country_code not in nominatims.keys():
                nomi = pgeocode.Nominatim(country_code)
                nominatims[country_code] = nomi
            else:
                nomi = nominatims[country_code]
        except Exception as e:
            num_bad_bois += 1
            postal_code = "17003"
            nomi = nominatims["US"] if "US" in nominatims.keys() else pgeocode.Nominatim("US")
        postal_code_info = nomi.query_postal_code(postal_code)
        if np.isnan(postal_code_info.latitude) or np.isnan(postal_code_info.longitude):
            coords.append([40.32927, -76.51553])
            num_bad_bois += 1
        else:
            coords.append([postal_code_info.latitude, postal_code_info.longitude])

    annville = [[40.32927, -76.51553]] * len(coords)
    distances_km = pgeocode.haversine_distance(annville, coords)

    avg_dist_km = sum(distances_km) / (len(distances_km) - num_bad_bois)
    distances_km = list(map(lambda d: avg_dist_km if d < 1.0 else d, distances_km))

    return data.assign(distance_to_lvc=distances_km)

# {'RW', 'VN', 'ZW', 'UZ', 'GN', 'AL', 'NG', 'NP', 'LB', 'KE', 'TW', 'ET', 'BS', 'GH'}
# {'ZW', 'KZ', 'SL', 'KP', 'JM', 'LR', 'NG', 'TT', 'CN', 'MA', 'LB', 'EG', 'SA', 'KE', 'AE', 'UZ', 'LS', 'GH', 'MN', 'ET'}
# {'CM', 'KE', 'GN', 'LR', 'UZ', 'TD', 'LB', 'ET', 'GH', 'KG', 'MU', 'HN', 'VN', 'ZW', 'RW', 'KZ', 'UG', 'NG', 'NP', 'SZ', 'SA', 'MA'}
