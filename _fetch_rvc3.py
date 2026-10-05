"""Explore RVC-Project repo to find model architecture code."""
import urllib.request
import json


BASE = "https://api.github.com/repos/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.read()
    except Exception as e:
        return None


def list_dir(path):
    data = fetch(BASE + "/contents/" + path)
    if data is None:
        return None
    return json.loads(data)


def walk(path, depth=0, max_depth=3):
    items = list_dir(path)
    if items is None:
        return
    for it in items:
        prefix = "  " * depth
        if it["type"] == "dir":
            print(f"{prefix}{it['name']}/")
            if depth < max_depth:
                walk(path + "/" + it["name"], depth + 1, max_depth)
        else:
            print(f"{prefix}{it['name']} ({it['size']} B)")


print("=== Top level ===")
walk("", 0, 2)
print("\n=== Search for SynthesizerTrn ===")
# Use grep API instead
url = BASE + "/search/code?q=class+SynthesizerTrn"
data = fetch(url)
if data:
    j = json.loads(data)
    if "items" in j:
        for item in j["items"][:10]:
            print(f"  {item['path']}")