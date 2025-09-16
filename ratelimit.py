# import requests
# url = "http://10.86.222.148:5839/identity/login"
# payload = {
#     "userName": "tanpt",
#     "password": "wrong",
#     "isUseBiometric": True,
#     "deviceId": "123",
#     "isGetMobileResponse": True
# }
# headers = {"Content-Type": "application/json"}
# for i in range(1000):
#     r = requests.post(url, json=payload, headers=headers)
#     print(i, r.status_code)


from locust import HttpUser, task, between

class LoginUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def login(self):
        self.client.post("http://10.86.222.148:5839/identity/login", json={
            "userName": "tanpt",
            "password": "123",
            "isUseBiometric": True,
            "deviceId": "123",
            "isGetMobileResponse": True
        })
