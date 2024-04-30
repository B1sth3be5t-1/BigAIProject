import pandas as pd
import numpy as np
import pycountry
import pgeocode


def load(filename):
    return pd.read_csv(filename)


def load_all():
    return pd.concat([load('data/fall-2022-noaddr.csv'), load('data/fall-2023-noaddr.csv'), load('data/9-2021-noaddr.csv')], ignore_index=True)


def create_key(df, cat):
    return {cat[x]: x-1 for x in range(len(pd.Categorical(df, categories=df.unique(), ordered=True).categories)+1)}


def make_categorical(df, column):
    df[column] =df[column].fillna(-1) #Handles N/A
    cat = list(pd.Categorical(df[column], categories=df[column].unique(), ordered=True).categories)
    cat.insert(0, "N/A")
    df[column] = pd.Categorical(df[column], categories=df[column].unique(), ordered=True).codes
    return df[column], create_key(df[column], cat)


def convert_to_ordinal(columns):
    df = load_all()
    for col in columns:
        df[col], key = make_categorical(df, col)
    return df, key


def convert_dates_to_days_before_may1(df, year):
    date_columns = ["APP_DATE", "INQUIRY_DATE", "APP_COMPLETE_DATE", "ON_SITE_VISIT_DATE1", "FIRSTSOURCE_DATE",
                    "COMM_1_DATE"]

    # TODO: Should we include file genesis date? The format is different in 2023. "FILE_GENESIS_DATE"

    # Convert date columns to datetime objects
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%Y%m%d')

    # Define the target date
    target_date = pd.to_datetime(f'{year}-05-01')

    # Calculate the number of days before May 1st for each date column
    for col in date_columns:
        df[col + "_M1"] = (target_date - df[col]).dt.days

    return df


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


# Function to take all visit columns, and count the max number of consecuitve Trues and Falses of each student
def visit_count(df):
    attended_columns = ['ON_SITE_VISIT_ATTENDED1', 'ON_SITE_VISIT_ATTENDED2','ON_SITE_VISIT_ATTENDED3', 'ON_SITE_VISIT_ATTENDED4', 'ON_SITE_VISIT_ATTENDED5',
                        'ON_SITE_VISIT_ATTENDED6', 'ON_SITE_VISIT_ATTENDED7', 'ON_SITE_VISIT_ATTENDED8',
                        'ON_SITE_VISIT_ATTENDED9', 'ON_SITE_VISIT_ATTENDED10']

    df['ON_SITE_TRUE_COUNT'] = 0
    df['ON_SITE_FALSE_COUNT'] = 0

    for index, row in df.iterrows():
        true_counts = [0]
        false_counts = [0]
        true_count = 0
        false_count = 0
        for col in attended_columns:
            if pd.notnull(row[col]):
                if row[col] == 'TRUE':
                    true_count += 1
                    false_counts.append(false_count)
                    false_count = 0
                elif row[col] == 'FALSE':
                    false_count += 1
                    true_counts.append(true_count)
                    true_count = 0
        true_counts.append(true_count)
        false_counts.append(false_count)
        true_max = max(true_counts)
        false_max = max(false_counts)
        df.at[index, 'ON_SITE_TRUE_COUNT'] = true_max
        df.at[index, 'ON_SITE_FALSE_COUNT'] = false_max

    return df


def visit_count2(df):
    date_columns = ['ON_SITE_VISIT_DATE1', 'ON_SITE_VISIT_DATE2', 'ON_SITE_VISIT_DATE3',
                    'ON_SITE_VISIT_DATE4', 'ON_SITE_VISIT_DATE5', 'ON_SITE_VISIT_DATE6',
                    'ON_SITE_VISIT_DATE7', 'ON_SITE_VISIT_DATE8', 'ON_SITE_VISIT_DATE9',
                    'ON_SITE_VISIT_DATE10']

    df['ON_SITE_VISIT_COUNT'] = 0

    for index, row in df.iterrows():
        count = sum(pd.notnull(row[col]) for col in date_columns)
        df.at[index, 'ON_SITE_VISIT_COUNT'] = count

    return df


def extra_curriculars(df):
    cols = ['COCURRIC_INTEREST1', 'COCURRIC_INTEREST2', 'COCURRIC_INTEREST3',
                        'COCURRIC_INTEREST4', 'COCURRIC_INTEREST5', 'COCURRIC_INTEREST6',
                        'COCURRIC_INTEREST7', 'COCURRIC_INTEREST8', 'COCURRIC_INTEREST9',
                        'COCURRIC_INTEREST10', 'COCURRIC_INTEREST11', 'COCURRIC_INTEREST12',
                        'COCURRIC_INTEREST13', 'COCURRIC_INTEREST14', 'COCURRIC_INTEREST15',
                        'COCURRIC_INTEREST16', 'COCURRIC_INTEREST17', 'COCURRIC_INTEREST18',
                        'COCURRIC_INTEREST19', 'COCURRIC_INTEREST20']

    df['MUSIC_INTRST'] = 0
    df['ATH_INTRST'] = 0

    athletics = {'football', 'soccer', 'sport', 'athletics', 'track',
              'volleyball', 'softball', 'basketball','lacrosse', 'cheerleading',
              'baseball', 'wrestling', 'cross country', 'golf', 'tennis','hockey',
            'field hockey', 'cross-country', 'indoor track', 'track and field',
                 'travel hockey', 'ice hockey', 'outdoor track', 'varsity wrestling',
                 'swim', 'swimming', 'dance', 'dancing', 'rugby', 'bowling'}

    music = {'music', 'band', 'theater', 'musical', 'choir', 'chorus','marching band', 'concert band',
             'jazz band', 'instrument','singing','music ensemble'}

    interest_count = []
    for index, row in df.iterrows():
        count = 0
        for col in cols:
            if pd.notnull(row[col]) and (row[col].lower() in music):
                df.at[index, 'MUSIC_INTRST'] = 1
                count += 1
                continue
            elif pd.notnull(row[col]) and (row[col].lower() in athletics):
                df.at[index, 'ATH_INTRST'] = 1
                count += 1
                continue
            elif pd.notnull(row[col]):
                count += 1
        interest_count.append(count)

    df['INTRST_COUNT'] = interest_count

    return df


def attended_count(df):
    attended_columns = ['ON_SITE_VISIT_ATTENDED1', 'ON_SITE_VISIT_ATTENDED2','ON_SITE_VISIT_ATTENDED3', 'ON_SITE_VISIT_ATTENDED4', 'ON_SITE_VISIT_ATTENDED5',
                        'ON_SITE_VISIT_ATTENDED6', 'ON_SITE_VISIT_ATTENDED7', 'ON_SITE_VISIT_ATTENDED8',
                        'ON_SITE_VISIT_ATTENDED9', 'ON_SITE_VISIT_ATTENDED10']

    df['ON_SITE_TRUE_COUNT'] = 0
    df['ON_SITE_FALSE_COUNT'] = 0

    for index, row in df.iterrows():
        true_counts = [0]
        false_counts = [0]
        true_count = 0
        false_count = 0
        for col in attended_columns:
            if pd.notnull(row[col]):
                if row[col] == 'TRUE':
                    true_count += 1
                    false_counts.append(false_count)
                    false_count = 0
                elif row[col] == 'FALSE':
                    false_count += 1
                    true_counts.append(true_count)
                    true_count = 0
        true_counts.append(true_count)
        false_counts.append(false_count)
        true_max = max(true_counts)
        false_max = max(false_counts)
        df.at[index, 'ON_SITE_TRUE_COUNT'] = true_max
        df.at[index, 'ON_SITE_FALSE_COUNT'] = false_max

    return df


def visits_attended(df):
    attended_columns = ['ON_SITE_VISIT_ATTENDED1', 'ON_SITE_VISIT_ATTENDED2','ON_SITE_VISIT_ATTENDED3', 'ON_SITE_VISIT_ATTENDED4', 'ON_SITE_VISIT_ATTENDED5',
                        'ON_SITE_VISIT_ATTENDED6', 'ON_SITE_VISIT_ATTENDED7', 'ON_SITE_VISIT_ATTENDED8',
                        'ON_SITE_VISIT_ATTENDED9', 'ON_SITE_VISIT_ATTENDED10']

    df['ON_SITE_TRUE_COUNT'] = 0
    df['ON_SITE_FALSE_COUNT'] = 0

    for index, row in df.iterrows():
        true_counts = [0]
        false_counts = [0]
        true_count = 0
        false_count = 0
        for col in attended_columns:
            if pd.notnull(row[col]):
                if row[col] == 'TRUE':
                    true_count += 1
                    false_counts.append(false_count)
                    false_count = 0
                elif row[col] == 'FALSE':
                    false_count += 1
                    true_counts.append(true_count)
                    true_count = 0
        true_counts.append(true_count)
        false_counts.append(false_count)
        true_max = max(true_counts)
        false_max = max(false_counts)
        df.at[index, 'ON_SITE_TRUE_COUNT'] = true_max
        df.at[index, 'ON_SITE_FALSE_COUNT'] = false_max

    return df


def remove_columns(data, label_list):
    data = data.drop(label_list, axis=1)
    return data

def_remove = ['STUDENT_ID_2', 'ZIP', 'SEARCH', 'DEPT_APP3_DEPARTMENT', 'DEPT_APP3_STATUS', 'BUMP_2017',
              'VIRTUAL_VISIT_TYPE2', 'VIRTUAL_VISIT_TYPE3', 'VIRTUAL_VISIT_TYPE4', 'VIRTUAL_VISIT_TYPE5', 'VIRTUAL_VISIT_TYPE6',
              'VIRTUAL_VISIT_TYPE7', 'VIRTUAL_VISIT_TYPE8', 'VIRTUAL_VISIT_TYPE9', 'VIRTUAL_VISIT_TYPE10', 'VIRTUAL_VISIT_DATE2',
              'VIRTUAL_VISIT_DATE3', 'VIRTUAL_VISIT_DATE4', 'VIRTUAL_VISIT_DATE5', 'VIRTUAL_VISIT_DATE6', 'VIRTUAL_VISIT_DATE7',
              'VIRTUAL_VISIT_DATE8', 'VIRTUAL_VISIT_DATE9', 'VIRTUAL_VISIT_DATE10', 'VIRTUAL_VISIT_ATTENDED1',
              'VIRTUAL_VISIT_ATTENDED2', 'VIRTUAL_VISIT_ATTENDED3', 'VIRTUAL_VISIT_ATTENDED4', 'VIRTUAL_VISIT_ATTENDED5',
              'VIRTUAL_VISIT_ATTENDED6', 'VIRTUAL_VISIT_ATTENDED7', 'VIRTUAL_VISIT_ATTENDED8', 'VIRTUAL_VISIT_ATTENDED9',
              'VIRTUAL_VISIT_ATTENDED10', 'ON_SITE_VISIT_ATTENDED1', 'ON_SITE_VISIT_ATTENDED2', 'ON_SITE_VISIT_ATTENDED3',
              'ON_SITE_VISIT_ATTENDED4', 'ON_SITE_VISIT_ATTENDED5', 'ON_SITE_VISIT_ATTENDED6', 'ON_SITE_VISIT_ATTENDED7',
              'ON_SITE_VISIT_ATTENDED8', 'ON_SITE_VISIT_ATTENDED9', 'ON_SITE_VISIT_ATTENDED10', 'FICE2', 'FICE3', 'FICE4', 'FICE5',
              'FICE6', 'FICE7', 'FICE8', 'FICE9', 'FICE10', 'FICE12', 'FICE13', 'FICE14', 'FICE15', 'FICE16', 'FICE17', 'FICE18', 'FICE19',
              'FICE20', 'COCURRICULAR1', 'COCURRICULAR2', 'COCURRICULAR3', 'COCURRICULAR4', 'COCURRICULAR5',
              'COCURRICULAR6', 'COCURRICULAR7', 'COCURRICULAR8', 'COCURRICULAR9', 'COCURRICULAR10', 'COCURRICULAR11',
              'COCURRICULAR12', 'COCURRICULAR13', 'COCURRICULAR14', 'COCURRICULAR15', 'COCURRICULAR16', 'COCURRICULAR17',
              'COCURRICULAR18', 'COCURRICULAR19', 'COCURRICULAR20', 'ON_SITE_VISIT_DATE1', 'ON_SITE_VISIT_DATE2', 'ON_SITE_VISIT_DATE3',
              'ON_SITE_VISIT_DATE4', 'ON_SITE_VISIT_DATE5', 'ON_SITE_VISIT_DATE6', 'ON_SITE_VISIT_DATE7',
              'ON_SITE_VISIT_DATE8', 'ON_SITE_VISIT_DATE9', 'ON_SITE_VISIT_DATE10', 'COCURRIC_INTEREST1',
              'COCURRIC_INTEREST2', 'COCURRIC_INTEREST3', 'COCURRIC_INTEREST4', 'COCURRIC_INTEREST5', 'COCURRIC_INTEREST6',
              'COCURRIC_INTEREST7', 'COCURRIC_INTEREST8', 'COCURRIC_INTEREST9', 'COCURRIC_INTEREST10', 'COCURRIC_INTEREST11',
              'COCURRIC_INTEREST12', 'COCURRIC_INTEREST13', 'COCURRIC_INTEREST14', 'COCURRIC_INTEREST15',
              'COCURRIC_INTEREST16', 'COCURRIC_INTEREST17', 'COCURRIC_INTEREST18', 'COCURRIC_INTEREST19',
              'COCURRIC_INTEREST20']
prob_remove = ['COUNTY', 'TERM_ENTERING', 'COMMUTER_ADMIT', 'INQUIRY_DATE', 'APP_DATE', 'APP_COMPLETE_DATE', 'ADMIT_DATE',
               'LEGACY_FAMILY_DESC', 'FIRST_GENERATION_CLIENT', 'ACAD_INTEREST1', 'DEP_APP1_DEPARTMENT2', 'DEPT_APP2_STATUS',
               'SportsAndRank', 'FICE1', 'FICE11']

not_remove = ['PRIMARY_STUDENT_ID', 'FILE_GENESIS_DATE', 'ADMIT_TYPE', 'INTERNATIONAL', 'CITIZEN_STATUS', 'FT_PT_IND', 'COMMUTER_APP', 'SEARCH_SOURCE',
              'APP_SOURCE', 'DEPOSIT_DATE', 'SEX', 'HISPANIC_IND', 'RACE', 'RELIGION', 'HOUSING_INTENT_APP', 'HS_PERCENTILE',
              'COUNSELOR_ASSIGNED', 'MAJOR_INTENDED1', 'MAJOR_INTENDED2', 'FIRSTSOURCE', 'FIRSTSOURCE_DATE', 'FIRST_FAFSA_FILE_DATE',
              'FIRST_FAFSA_RECEIVED_DATE', 'INITIAL_NEED_BASED_PACKAGE_MAIL_DATE', 'COST_OF_ATTENDANCE', 'TUITION_FOR_DISCOUNT',
              'TUITION_FEES', 'COMPREHENSIVE_FEE', 'BOOKS', 'TRANSPORTATION', 'ROOM_BOARD_ON_CAMPUS', 'MISC_PERSONAL_EXPENSES',
              'LOAN_ORIGINATION_FEES', 'FM_PARENT_AGI', 'FM_STUDENT_AGI', 'FM_FAMILY_INCOME', 'FM_FAMILY_SIZE', 'FM_NUMBER_EARNERS',
              'FM_NUMBER_IN_COLLEGE', 'FM_ASSETS', 'FM_CASH', 'FM_INVESTMENT_NETWORTH', 'FM_BUSINESS_FARM', 'FM_EFC', 'FM_EFC_PARENT',
              'FM_EFC_STUDENT', 'FM_DEP_IND', 'FM_PARENT1_ED_ATT', 'FM_PARENT2_ED_ATT', 'FM_MARITAL_STATUS', 'TOTAL_INST_GRANT',
              'TOTAL_GRANT', 'OTHER_FEDERAL_GRANT', 'TOTAL_FEDERAL_GRANT', 'TOTAL_STATE_GRANT', 'TOTAL_OUTSIDE_PRIVATE_GRANT',
              'TUITION_REMIISION', 'TUITION_REMIISION_AMOUNT', 'INITIAL_INTENDED_MAJOR1', 'DEP_APP1_DEPARTMENT', 'DEPT_APP1_STATUS',
              'Unique_Visit_Days_Count', 'Unique_Visit_Days_NoShow_Count', 'Virtual_Visit_Count', 'VIRTUAL_VISIT_DATE1']
unknown_remove = ['COUNTRY', 'APPLICANT_TYPE', 'DEFER_APPLICANT_DECISION', 'COND_ACCEPT_DATE', 'COND_ACCEPT_REASON',
                  'WITHDRAWN_DATE', 'DENIED_DATE', 'ENROLLED_IND', 'BIRTHDATE', 'URM', 'HIGH_SCHOOL_CEEB', 'HS_NAME', 'TRANSFER_CEEB',
                  'TRANSFER_INST_NAME', 'HS_GPA_SCALE', 'HS_ORIG_GPA', 'HS_GPA_SCRUBBED', 'HS_RANK', 'HS_SIZE', 'NEED_ANALYSIS_USED',
                  'EFC_USED_FOR_PACKAGING', 'VERICATION_SELECTED', 'VERIFICATION_COMPLETE', 'IGRANT_FOR_DISCOUNT',
                  'INST_NON_NEED_MONEY_FOR_MERIT', 'INST_NON_NEED_MONEY_FOR_MERIT_DATE', 'PELL', 'SEOG', 'WORK_AMOUNT',
                  'LOAN_AMOUNT_SUB', 'UNSUB', 'LOAN_AMOUNT_PERKINS', 'LOAN_AMOUNT_STUDENT_PRIVATE_ALTERNATIVE',
                  'LOAN_AMOUNT_PARENT_PLUS', 'TUITION_EXCHANGE', 'EXCHANGE_AMOUNT', 'APPEAL_FLAG1', 'APPEAL_DATE1', 'APPEAL_AMOUNT1',
                  'APPEAL_OUTCOME_DESC1', 'FINAL_ENROLLED_IND', 'SCHOLARSHIP_EVEN_IF_NOT_PACKAGED',
                  'TUITION_EVEN_IF_NOT_PACKAGED', 'FA_INTENT_NEED', 'CALC_SCHOLARSHIP_AMOUNT', 'SCHOLARSHIP_SAT_ISSUE_FALL_2017',
                  'INITIAL_INTENED_MAJOR2', 'Exclude_Tests', 'WILL_NOT_FILE', 'APPEAL_OUTCOME_CODE1',
                  'LOAN_AMOUNT_PARENT_PRIVATE_ALTERNATIVE', 'Scholarship_Override__c', 'CALC_ACRK', 'CALC_NDRK']



# takes a single column name and its keys and writes them to a new csv file
def write_keys(key, column):
    header = [column, 'num_key']
    d = pandas.DataFrame(key, header)
    d.to_csv('../data/' + column + '_key.csv')








df = pd.read_csv("9-2021-noaddr.csv", dtype=object)

# Select  the columns of visits
visit_columns = [f'ON_SITE_VISIT_ATTENDED{i}' for i in range(1, 11)]
visit_df = df[visit_columns]

# Print the selected columns, to validate
print(visit_df.head(10))


df = pd.read_csv("9-2021-noaddr.csv", dtype=object)

df = visit_count(df)
print(df[['ON_SITE_TRUE_COUNT', 'ON_SITE_FALSE_COUNT']].head(50))

#
#
################################################################################
#
#


columns = ["MAJOR_INTENDED1"]
df, key = convert_to_ordinal(["MAJOR_INTENDED1"])
print(df, key)


#
#
################################################################################
#
#



# Read the CSV into a DataFrame
df_2021 = pd.read_csv('9-2021-noaddr.csv')
df_2022 = pd.read_csv('fall-2022-noaddr.csv')
df_2023 = pd.read_csv('fall-2023-noaddr.csv')

print("2021 Columns:")
print(df_2021.columns)

print("2022 Columns:")
print(df_2022.columns)

print("2023 Columns:")
print(df_2023.columns)

# Call the function to convert dates to days before May 1st
new_df_2021 = convert_dates_to_days_before_may1(df_2021, 2021)
new_df_2022 = convert_dates_to_days_before_may1(df_2022, 2022)
new_df_2023 = convert_dates_to_days_before_may1(df_2023, 2023)

# Print the modified DataFrame
print(new_df_2021)
print(new_df_2022)
print(new_df_2023)


#
#
################################################################################
#
#



df = pd.read_csv("test.csv", dtype=object)

# Select  the columns of visits
visit_columns = [f'ON_SITE_VISIT_ATTENDED{i}' for i in range(1, 11)]
visit_df = df[visit_columns]

# Print the selected columns, to validate
# print(visit_df.head(10))

# Function to take all visit columns, and coutn the max number of consecuitve Trues and Falses of each student




df = attended_count(df)
print(df[['ON_SITE_TRUE_COUNT', 'ON_SITE_FALSE_COUNT']].head(50))


#
#
################################################################################
#
#


data = pd.read_csv("CSV_Files/fall-2021-noaddr.csv")



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


#
#
################################################################################
#
#



df = pd.read_csv("test.csv", dtype=object)

df = visit_count(df)
# print(df['ON_SITE_VISIT_COUNT'].head(30))
#
df = extra_curriculars(df)
# print(df['MUSIC_INTRST'].head(5))
# print(df['ATH_INTRST'].head(5))
# print(df['INTRST_COUNT'].head(5))
df = visits_attended(df)


#
#
################################################################################
#
#



# Read the CSV into a DataFrame
df_2021 = pd.read_csv('9-2021-noaddr.csv')
df_2022 = pd.read_csv('fall-2022-noaddr.csv')
df_2023 = pd.read_csv('fall-2023-noaddr.csv')

print("2021 Columns:")
print(df_2021.columns)

print("2022 Columns:")
print(df_2022.columns)

print("2023 Columns:")
print(df_2023.columns)

# Call the function to convert dates to days before May 1st
new_df_2021 = convert_dates_to_days_before_may1(df_2021, 2021)
new_df_2022 = convert_dates_to_days_before_may1(df_2022, 2022)
new_df_2023 = convert_dates_to_days_before_may1(df_2023, 2023)

# Print the modified DataFrame
print(new_df_2021)
print(new_df_2022)
print(new_df_2023)
