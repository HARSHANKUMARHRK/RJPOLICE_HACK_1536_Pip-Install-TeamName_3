import requests
import pandas as pd

def get_ip_address():
    url = 'https://api.ipify.org'
    response = requests.get(url)
    ip_address = response.text
    return ip_address

def get_location_from_ip(ip_address):
    access_token = 'ca2235bc0acff2'  
    url = f'https://ipinfo.io/{ip_address}?token={access_token}'
    
    response = requests.get(url)
    data = response.json()

    city = data.get('city')
    region = data.get('region')
    country = data.get('country')
    loc = data.get('loc') 
    latitude, longitude = loc.split(',') if loc else (None, None)
    location = {
        'city': city,
        'region': region,
        'country': country,
        'latitude': latitude,
        'longitude': longitude
    }

    df = pd.read_csv("police.csv")

    # Convert latitude and longitude to floating-point numbers
    given_latitude = float(location["latitude"]) if location["latitude"] else None
    given_longitude = float(location["longitude"]) if location["longitude"] else None

    matching_rows = df[(df["Latitude"] == float(34.9540542)) & (df["Longitude"] == float(135.7515062))]
    
    if not matching_rows.empty:
        police_station_name = matching_rows["Police Station Name"].iloc[0]
        print(police_station_name)
    
    location_str = f"{city}, {region}, {country}"
    return location_str

ip = get_ip_address()
loc = get_location_from_ip(ip)
