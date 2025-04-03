import requests

response = requests.get(
    "https://api.electricitymap.org/v3/power-breakdown/past-range?start=2017-01-01&end=2017-01-9&zone=SE-SE3&resolution=60",
    headers={
        "auth-token": f"kpAMSQ6VQE9SO"
    }
)
print(response.json())
# import requests

# response = requests.get(
#     "https://api.electricitymap.org/v3/power-breakdown/past?zone=SE-SE3&resolution=60&datetime=2017-01-01",
#     headers={
#         "auth-token": f"kpAMSQ6VQE9SO"
#     }
# )
# print(response.json())