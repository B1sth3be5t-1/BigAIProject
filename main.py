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

    # Take each country and map them all to strings
    country_col = country_col.map(lambda c: str(c))

    # create and compile a list of pycountry countries
    new_countries = []
    for c in country_col:
        # if the country field is null, that means the applicant was in the US. The fields in this column are blank
        if c == "nan":
            country = pycountry.countries.get(alpha_2="us")
        # some of the country lengths are 2 characters long. We'll plug this value directly into the search for 2
        # digit country codes
        elif len(c) == 2:
            country = pycountry.countries.get(alpha_2=c)
        else:
            # so turkey gives us problems. We have to change it to the official name
            if c == "Turkey":
                c = "turkiye"
            # if c is not Turkey, we can fuzzy search for the country, which returns a list, so we pull the first
            # item (the best choice) from the list
            country = pycountry.countries.search_fuzzy(c)[0]
        # append the 2 digit country code to new_countries
        new_countries.append(country.alpha_2)

    # for s in set(new_countries):
    #    print(s)

    # create a blank list of coordinates, pgeocode nominatims, and the number of countries that are being an issue
    coords = []
    nominatims = {}
    num_bad_bois = 0

    # loop through the 2 digit country codes and postal codes that we have so far
    for country_code, postal_code in zip(new_countries, stripped_zips):
        # we need a try block here because sometimes pgeocode.Nominatim fails if the country isn't supported
        try:
            # Use some dynamic programming to see if we've already hit the API
            if country_code not in nominatims.keys():
                # find the country by hitting the API
                nomi = pgeocode.Nominatim(country_code)
                # update our list to add the country based on its country code
                nominatims[country_code] = nomi
            else:
                # if nomi exists in the set, use its value instead of hitting the API. Speeds up a lot
                nomi = nominatims[country_code]
        # if pgeocode.Nominatim fails, meaning the country isn't supported by the Pgeocode Library. This typically
        # happens for smaller countries
        except Exception as e:
            # increment number of problem countries
            num_bad_bois += 1

            # set the postal code to Annville PA, so we can get the distance to be 0 after haversine
            postal_code = "17003"

            # if us has already been searched for, pull it from nominatims, else get the US pgeocode nominatim
            nomi = nominatims["US"] if "US" in nominatims.keys() else pgeocode.Nominatim("US")

        # get postal code information for the pgeocode nominatim object
        postal_code_info = nomi.query_postal_code(postal_code)

        # if the postal code lat or long is nan, then we add Annville's lat and long information (and add another
        # count to the bad country records)
        if np.isnan(postal_code_info.latitude) or np.isnan(postal_code_info.longitude):
            coords.append([40.32927, -76.51553])
            num_bad_bois += 1
        else:
            # append to coords the postal code's lat and long
            coords.append([postal_code_info.latitude, postal_code_info.longitude])

    # create a list of annville's lat and long, duplicated times the length of coords
    annville = [[40.32927, -76.51553]] * len(coords)
    # Get our list of distances. The haversine distance function can take lists of pairs and do them pairwise
    distances_km = pgeocode.haversine_distance(annville, coords)

    # Get the average distance by summing all the distances and divide by the total number of rows - the number of
    # bad rows to get an accurate average for our good distances
    avg_dist_km = sum(distances_km) / (len(distances_km) - num_bad_bois)
    # map any of the distances that are less than 1km (aka we set the bad records to be located in Annville) to the
    # average distance of the good distances
    distances_km = list(map(lambda d: avg_dist_km if d < 1.0 else d, distances_km))

    # tack on the column to the inputted dataframe
    return data.assign(distance_to_lvc=distances_km)

# among the list of 3 datasets, here are all the countries that are unsupported by Pgeocode. We couldn't find any
# library that was better though

# {'RW', 'VN', 'ZW', 'UZ', 'GN', 'AL', 'NG', 'NP', 'LB', 'KE', 'TW', 'ET', 'BS',
# 'GH'}

# {'ZW', 'KZ', 'SL', 'KP', 'JM', 'LR', 'NG', 'TT', 'CN', 'MA', 'LB', 'EG', 'SA', 'KE', 'AE', 'UZ', 'LS', 'GH',
# 'MN', 'ET'}

# {'CM', 'KE', 'GN', 'LR', 'UZ', 'TD', 'LB', 'ET', 'GH', 'KG', 'MU', 'HN', 'VN', 'ZW', 'RW', 'KZ', 'UG',
# 'NG', 'NP', 'SZ', 'SA', 'MA'}
######################################################################################################################
# unique set of countries not supported by PGeocode: {'KG', 'CN', 'GN', 'LS', 'RW', 'UZ', 'NP', 'EG', 'GH', 'HN',
# 'UG', 'CM', 'NG', 'TT', 'JM', 'LR', 'ZW', 'TD', 'MU', 'KP', 'SL', 'KZ', 'SZ', 'LB', 'AL', 'TW', 'BS', 'AE', 'MN',
# 'VN', 'SA', 'KE', 'ET', 'MA'}
