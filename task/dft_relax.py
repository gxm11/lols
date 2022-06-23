from database import prepare_task
import model.aims as aims
import model.convert as convert
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

with open("%s/%s-%d-%d.xyz" % (folder, material, iteration, key), "w") as f:
    e, pos = aims.final_structure(folder)
    text = convert.generate_xyz(e, pos, material)
    f.write(text)

is_converged = aims.is_converged(folder)
is_finished = aims.is_finished(folder)

if is_finished:
    energy = aims.final_energy(folder) - phys.Zero_Energy
else:
    energy = None

all_energies, _force = aims.all_energies_and_forces(folder)
init_energy = all_energies[0] - phys.Zero_Energy

data = {
    "init_vector": vector,
    "init_energy": init_energy,
    "finish": is_finished,
    "converge": is_converged,
    "energy": energy
}

with open("%s/result.json" % folder, "w") as f:
    json.dump(data, f)

log.info("converged: %s after %d steps, energy: %f -> %f" %
         (is_converged, len(all_energies) - 1, init_energy, energy))
