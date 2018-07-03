#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018  <@DESKTOP-TA60DPH>
#
# Distributed under terms of the MIT license.
# Auther : RB

import requests 

import simplejson as json
from simplejson.compat import StringIO
"""
get the ranking between NTU on kaggle competition
"""
rank = requests.get("https://www.kaggle.com/c/9489/leaderboard.json?includeBeforeUser=false&includeAfterUser=true")

io = StringIO(rank.text)
data = json.load(io)

before = data["beforeUser"]
near = data["nearUser"]
after = data["afterUser"]

counter = 1
print("   ",format("rank","^6"),format('score',"^15"),format('teamName',"<30"))
position = 0
for i in before:
    our = ' '
    if i['teamName'][:4] == "NTU_":
        if i['teamName'] == "NTU_r06942018___":
            our = '*'
            position = counter
        print('['+our+']',format(counter,"^6"),format(i['score'],"^15"),format(i['teamName'],"<30"))
        counter+=1
for i in near:
    our = ' '
    if i['teamName'][:4] == "NTU_":
        if i['teamName'] == "NTU_r06942018___":
            our = '*'
            position = counter
        print('['+our+']',format(counter,"^6"),format(i['score'],"^15"),format(i['teamName'],"<30"))
        counter+=1

for i in after:
    our = ' '
    if i['teamName'][:4] == "NTU_":
        if i['teamName'] == "NTU_r06942018___":
            our = '*'
            position = counter
        print('['+our+']',format(counter,"^6"),format(i['score'],"^15"),format(i['teamName'],"<30"))
        counter+=1

if position < 10:
    score = 2
if position < 8:
    score = 5
if position < 6:
    score = 7.5 
if position < 4:
    score = 10

print("")
print("Rank (in NTU) : ",position,"/ 10  ") 
print("Ranking Point : ",score)
print("")



