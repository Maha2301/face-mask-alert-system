import requests
url = "https://www.fast2sms.com/dev/bulk"
payload = "sender_id=FSTSMS&message=test&language=english&route=p&numbers=9894385585"
headers = {
'authorization': "3ksNM91glvmZFV2jOJaQSoWifKP6hbn4HzEA0Gpy8u5RDTcLtdCzJ0rBNtT1DXVpR2yc9lgWYSAGhwI4",
'Content-Type': "application/x-www-form-urlencoded",
'Cache-Control': "no-cache",
}
response = requests.request("POST", url, data=payload, headers=headers)
print(response.text)