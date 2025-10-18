from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)  # allows frontend JS to access backend

def validate_ssl_certificate(host):
    """Check if SSL is valid by sending an HTTPS GET request"""
    try:
        response = requests.get(f"https://{host}", verify=True, timeout=5)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False

@app.route("/scan", methods=["POST"])
def scan_url():
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    parsed_url = urlparse(url if url.startswith("http") else "https://" + url)
    protocol = parsed_url.scheme
    domain = parsed_url.netloc
    url_length = len(url)
    ssl_valid = validate_ssl_certificate(domain)

    result = {
        "url": url,
        "protocol": protocol,
        "domain": domain,
        "url_length": url_length,
        "ssl_valid": ssl_valid,
    }
    return jsonify(result)

if __name__ == "__main__":
    print("✅ Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
