from clean_data import *

# get all 3 years worth of csv's

data2021 = load("CSV_Files/fall-2021-noaddr.csv")
data2022 = load("CSV_Files/fall-2022-noaddr.csv")
data2023 = load("CSV_Files/fall-2023-noaddr.csv")

# makes 3 files with the converted days before May 1
data2021 = convert_dates_to_days_before_may1(data2021, 2021)
data2022 = convert_dates_to_days_before_may1(data2022, 2022)
data2023 = convert_dates_to_days_before_may1(data2023, 2023)

# the 2022 and 23 data sets have a first_gen_any_source column that 2021 doesn't have, so we have to remove it
data2022.drop('First_Gen_Any_Source', axis=1, inplace=True)
data2023.drop('First_Gen_Any_Source', axis=1, inplace=True)

# now we stitch the datasets together into one big file
df = pd.concat([data2021, data2022, data2023], axis=0)

# create a list of the visit columns
visit_columns = [f'ON_SITE_VISIT_ATTENDED{i}' for i in range(1, 11)]
visit_df = df[visit_columns]

# get number of visits
df = visits_attended(df)

# # get distances from Annville to zip codes
# df = getDistances(df)

# set majors for students
columns = ["MAJOR_INTENDED1"]
df, key = convert_to_ordinal(df, ["MAJOR_INTENDED1"])

# get counts of extracurriculars
df = extra_curriculars(df)

# aggregate total scholarship amount
df = sum_scholarship(df)

df = remove_columns(df, def_remove)
df = remove_columns(df, prob_remove)
df = remove_columns(df, unknown_remove)

# writes the data to an Excel file, easier to filter and view this way
datatoExcel = pd.ExcelWriter('allRows.xlsx')
df.to_excel(datatoExcel)
datatoExcel.close()