import requests
import pandas as pd
def get_ip_address():
    url = 'https://api.ipify.org'
    response = requests.get(url)
    ip_address = response.text
    return ip_address

def get_location_from_ip(ip_address):
    access_token = 'ca2235bc0acff2'  # Replace with your actual access token from ipinfo.io
    url = f'https://ipinfo.io/{ip_address}?token={access_token}'
    
    response = requests.get(url)
    data = response.json()

    city = data.get('city')
    region = data.get('region')
    country = data.get('country')


    df= pd.read_csv("police.csv")
    if(df["Latitude"]==)
    
    location = f"{city}, {region}, {country}"
    return location



ip=get_ip_address()
loc=get_location_from_ip(ip)