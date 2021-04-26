import json
import sys
import os
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

N = int(sys.argv[1])
a = float(sys.argv[2])
b = float(sys.argv[3])
use_sto = int(sys.argv[4])


param = {
    "N": N,
    "alpha": 3,
    "a": a,
    "b": b,
    "use_sto": use_sto,
    "optimizer": {
        "name": "SGD",
        "alpha": 0.02,
        "p": 0.0
    },
    "number_of_temperatures": 16,
    "number_of_chains_per_each": 1
}

print(json.dumps(param, indent=4))
