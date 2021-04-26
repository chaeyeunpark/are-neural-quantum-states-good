import json
import sys
import os
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

N = int(sys.argv[1])
delta = float(sys.argv[2])
sign_rule = (sys.argv[3].lower() == 'true')

param = {
    "N": N,
    "alpha": 3,
    "delta": delta,
    "sign_rule": sign_rule,
    "optimizer": {
        "name": "SGD",
        "alpha": 0.02,
        "p": 0.0
    },
    "number_of_temperatures": 16,
    "number_of_chains_per_each": 1
}

print(json.dumps(param, indent=4))
