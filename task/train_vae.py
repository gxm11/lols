from database import prepare_task, get_task_id
import numpy as np
import model.vae as vae
import model.log as log
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)


def vae_input_data(data, alpha):
    e = data[:, -1]
    # resize energy range and select low energy parts
    e = np.array([np.log(1 + x) if x > 0 else x for x in e])
    mean, std = np.mean(e), np.std(e)
    upper_limit = mean + std * alpha
    log.info(" - mean = %f, std = %f, upper_limit = %f" %
             (mean, std, upper_limit))
    ignore_args = np.argwhere(e >= upper_limit).reshape(-1)
    log.info(" - ignore %d high-energy data" % ignore_args.size)
    data[:, -1] = e
    data = np.delete(data, ignore_args, axis=0)
    return data


task_id = get_task_id("data_generation", iteration)
data = np.loadtxt("work/%d/data_pool.dat" % task_id)
data = vae_input_data(data, alpha=config["cutoff"])
np.savetxt("%s/input_data.dat" % folder, data)

config["config"]["input_data"] = "%s/input_data.dat" % folder
vae.setup(config["config"], folder)
vae.run(folder, config["execute"], config["epoches"])
