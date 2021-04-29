# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:10:12 2021

@author: Administrator
"""
from twilio.rest import Client
account_sid = "ACe368ef538ae07c1644c0fad7ceac2928"
auth_token = "ca0581d49ec31ab308e0506d7b37cb08"
client = Client(account_sid, auth_token)
client.messages.create(from_ = "+12565768150", body = "A person has violated facial mask policy. Kindly check the camera to recognise the person", to="+91 98943 85585")