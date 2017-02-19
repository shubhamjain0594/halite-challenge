import simplejson as json
import requests
import urllib
import numpy as np
from time import sleep

url = "https://halite.io/api/web/game?userID=2609&limit=500"
r = requests.get(url)
data = r.json()

testfile = urllib.request.URLopener()

for datum in data:
    game_id = datum["replayName"]
    print(game_id)
    request = "https://s3.amazonaws.com/halitereplaybucket/{}".format(game_id)
    testfile.retrieve(request, "{}.gzip".format(game_id))
    sleep(max(0, np.random.randn()))
