# pip install pgeocode
# OR
# add pgeocode to requirements.txt
import pgeocode

nomi = pgeocode.Nominatim('ca')
print(nomi.query_postal_code('T6J 2P2'))

dist = pgeocode.GeoDistance('ca')
applicant_zip = 'T6J 2P2'
dist_miles = dist.query_postal_code('17003', applicant_zip) / 1.609
print(dist_miles)
