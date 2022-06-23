from database import prepare_task, get_task_id
import model.vae as vae
import numpy as np
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)


def random_samples(n, dataset, expand):
    _min = np.min(dataset, axis=0).reshape(1, -1)
    _max = np.max(dataset, axis=0).reshape(1, -1)
    dim = dataset.shape[1]
    r = np.random.random((n, dim)) * (1 + expand) - expand * 0.5
    return _min + (_max - _min) * r


task_id = get_task_id("train_vae", iteration)
mu = np.loadtxt("work/%d/mu.dat" % task_id)
config["config"]["state_dict"] = "work/%d/vae.pt" % task_id

vae.setup(config["config"], folder)
sample = random_samples(config["number"], mu, config["expand"])
recon = vae.low2high(sample)
np.savetxt("%s/sample.dat" % folder, sample)
np.savetxt("%s/recon.dat" % folder, recon)
