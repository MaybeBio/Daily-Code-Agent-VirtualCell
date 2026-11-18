import sys
import os
# Add the BrainTwin package to the import path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "braintwin")))
BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "results"))
os.makedirs(OUT, exist_ok=True)
import json
from tqdm import tqdm
from evaluation_utils import SCQ_metrics

from multiagent_system.main import create_system
MAS = create_system()

system_ready = MAS.test_system()

# single_cell_DEG
with open(os.path.join(BENCH, "singlecell_DEG.json"), "r") as f:
   questions = json.load(f)
print(len(questions))

result_list = []
for question in tqdm(questions):
   result_list.append(MAS.query(question['input']))

with open(os.path.join(OUT, "singlecell_DEG.json"), "w") as f:
   json.dump(result_list,f)

# single_cell_region
with open(os.path.join(BENCH, "singlecell_region.json"), "r") as f:
   questions = json.load(f)
print(len(questions))

result_list = []
for question in tqdm(questions):
   result_list.append(MAS.query(question['input']))

with open(os.path.join(OUT, "singlecell_region.json"), "w") as f:
   json.dump(result_list,f)

# genomics
with open(os.path.join(BENCH, "genomics_SCQ.json"), "r") as f:
    questions = json.load(f)
print(len(questions))

result_list = []
for question in tqdm(questions):
    result_list.append(MAS.query(question['input']))

with open(os.path.join(OUT, "genomics_V6.json"), "w") as f:
    json.dump(result_list,f)

# image
with open(os.path.join(BENCH, "image_5K.json"), "r") as f:
    questions = json.load(f)
print(len(questions))

result_list = []
for question in tqdm(questions):
    result_list.append(MAS.query(question['input']))

with open(os.path.join(OUT, "image.json"), "w") as f:
    json.dump(result_list,f)