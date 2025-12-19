import requests

# Test with specific filters
url = "http://127.0.0.1:8000/api/topics/search"
data = {
    "query": "agriculture",
    "department": "Computer Engineering", 
    "option": "SEN",
    "year": 2020,
    "status": "Taken",
    "limit": 5
}

response = requests.post(url, json=data)
print("Status code:", response.status_code)
result = response.json()
print("Total results:", result.get("total_results"))
print("Number of topics:", len(result.get("topics", [])))

if result.get("topics"):
    print("\nTopics found:")
    for i, topic in enumerate(result["topics"]):
        print(f"{i+1}. {topic.get('title', 'No title')}")
        print(f"   Department: {topic.get('department')}")
        print(f"   Option: {topic.get('option')}")
        print(f"   Year: {topic.get('year')}")
        print()
else:
    print("No topics found")