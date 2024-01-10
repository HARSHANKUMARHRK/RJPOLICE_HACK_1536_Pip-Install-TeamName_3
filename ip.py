import requests

def get_location_from_ip(ip_address):
    access_token = 'ca2235bc0acff2'  # Replace with your actual access token from ipinfo.io
    url = f'https://ipinfo.io/{ip_address}?token={access_token}'
    
    response = requests.get(url)
    data = response.json()
    
    # Extract relevant information
    city = data.get('city')
    region = data.get('region')
    country = data.get('country')
    
    location = f"{city}, {region}, {country}"
    return location

# Example usage:
user_ip = input("Enter the IP address: ")  # Enter the user's IP address here
location = get_location_from_ip(user_ip)
print(f"The location of {user_ip} is: {location}")
