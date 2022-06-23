from database import prepare_task
import model.aims as aims
import model.log as log
import importlib
import json
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

material = config["material"]
phys = importlib.import_module("lib.material.%s.physics" % material)
vector = config["vector"]
atoms = phys.feature2atoms(vector)
aims.setup(config["config"], atoms, folder)
aims.run(folder, config["aims_execute"])

if aims.is_finished(folder):
    energy = aims.final_energy(folder) - phys.Zero_Energy
else:
    energy = None

log.info("vector: %s" % vector)
log.info("energy: %f" % energy)

with open("%s/result.json" % folder, "w") as f:
    json.dump({"vector": vector, "energy": energy}, f)
