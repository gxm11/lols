from database import get_task_id, prepare_task
import model.boss as boss
import model.log as log
import model.meanshift as meanshift
import numpy as np
import json
import sys

from scipy.optimize import minimize

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

task_id = get_task_id("gp_model", iteration)
gp_data = np.loadtxt("work/%d/gp_data.dat" % task_id)
gp_hypers = json.load(open("work/%d/gp_hypers.json" % task_id))
period = config.get("period", None)
error = config["error"]

model, Y_mean = boss.build_gpymodel(config["config"], gp_data, gp_hypers)
data_X = gp_data[:, 0:-1]


def f(a):
    value, uncert = model.predict(a.reshape(1, -1))
    return value[0][0]


def get_localminimum(x_start):
    _x = x_start
    for _i in range(3):
        ret = minimize(f, _x, method="CG")
        if period is None:
            break
        _x = ret.x
        if np.max(_x) < period / 2 and np.min(_x) > - period / 2:
            break
        _x = (_x + period / 2) % period - period / 2
    return ret


xs = []
for i, row in enumerate(data_X):
    ret = get_localminimum(row)
    log.info(" - worker %d / %d done." % (i + 1, data_X.shape[0]))
    if ret.success:
        xs.append(ret.x)

if period is not None:
    xs = (np.array(xs) + period / 2) % period - period / 2
else:
    r = config["space_pad"]
    data_X_min = np.min(data_X, axis=0)
    data_X_max = np.max(data_X, axis=0)
    _xmin = data_X_min * (1 + r) + data_X_max * (-r)
    _xmax = data_X_min * (-r) + data_X_max * (1 + r)
    new_xs = []
    for i, x in enumerate(xs):
        if np.all(x > _xmin) and np.all(x < _xmax):
            new_xs.append(x)
    xs = np.array(new_xs)

ys = model.predict(xs)

gp_minimum = np.hstack((xs, ys[0], ys[1]))
log.info("minimum size = %d" % xs.shape[0])
np.savetxt("%s/gp_minimum_raw.dat" % folder, gp_minimum)

centers, _ = meanshift.merge_centers_with_values(
    gp_minimum[:, 0:-2], gp_minimum[:, -2], error=error, periodic=period)
log.info("centers size = %d" % centers.shape[0])
np.savetxt("%s/gp_minimum.dat" % folder, centers)
