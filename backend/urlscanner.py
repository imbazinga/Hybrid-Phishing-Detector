import requests
from urllib.parse import urlparse

def validate_ssl_certificate(host):
    try:
        response = requests.get(f"https://{host}", verify=True)
        response.raise_for_status()  # Check if the response is successful (status code 2xx)
        print(f"SSL certificate for {host} is valid.")
    except requests.exceptions.RequestException as e:
        print(f"Error validating SSL certificate: {e}")

def scan_url(url):
    try:
        parsed_url = urlparse(url)
        protocol = parsed_url.scheme
        url_length = len(url)
        domain = parsed_url.netloc

        print(f"URL: {url}")
        print(f"Protocol: {protocol}")
        print(f"URL Length: {url_length} characters")
        print(f"Domain: {domain}")

        validate_ssl_certificate(parsed_url.netloc)

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    url = input("Enter a URL to scan: ")
    scan_url(url)

