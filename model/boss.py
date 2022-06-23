# ---------------------------------------------------------
# Bayesian optimization model
# ---------------------------------------------------------
import model.log as log
import os
import subprocess
import numpy as np
import json


def setup(config, data, folder):
    log.info("setup boss...")
    if not os.path.exists(folder):
        os.mkdir(folder)
    with open("%s/config.json" % folder, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    content = generate_bossin(data, config)
    with open("%s/boss.in" % folder, "w") as f:
        f.write(content)

    return folder


def run(workdir, execute, options="s"):
    log.info("run boss at <%s>, options <%s> ..." % (workdir, options))
    cmd = execute.split()
    if options == "o":
        cmd += ["o", "boss.in"]
    if options == "s":
        cmd += ["s", "boss.in"]
    if options == "op":
        cmd += ["op", "boss.in"]
    if options == "p":
        cmd += ["p", "boss.rst", "boss.out"]
    if options == "or":
        cmd += ["o", "boss.rst"]
    if options == "opr":
        cmd += ["op", "boss.rst"]

    subprocess.call(cmd, cwd=workdir)

    b = subprocess.run(["grep", "Have a nice day", "boss.out"],
                       cwd=workdir, stdout=subprocess.DEVNULL)
    if b.returncode != 0:
        log.error("Error happends on running boss, workdir: <%s>!" % workdir)
        return False
    else:
        return True


def local_minima(workdir):
    fd = "%s/postprocessing/data_local_minima" % workdir
    if not os.path.exists(fd):
        raise log.error("please run boss op before get localminima")

    fn = os.listdir(fd)[0]
    return np.loadtxt("%s/%s" % (fd, fn), ndmin=2)


def hyper_parameters(workdir):
    try:
        boss_out = "%s/boss.out" % workdir
        d = {}
        d["gp_parameters"] = check_bossout(
            "GP model hyperparameters", boss_out)
        d["next_sampling"] = check_bossout("Next sampling location", boss_out)
        d["global_minimum"] = check_bossout(
            "Global minimum prediction", boss_out)
        d["best_acquisition"] = check_bossout("Best acquisition", boss_out)
        return d
    except:
        return None


def build_gpymodel(config, data, hypers):
    import GPy
    noise, kernel = config["noise"], config["kernel"]
    # configs
    args = {
        "input_dim": data.shape[1] - 1,
        "variance": hypers[-1],
        "lengthscale": hypers[0:-1],
    }
    # setup kernels
    if kernel == "rbf":
        kernel = GPy.kern.RBF(ARD=True, **args)
    if kernel == "mat32":
        kernel = GPy.kern.Matern32(ARD=True, **args)
    if kernel == "mat52":
        kernel = GPy.kern.Matern52(ARD=True, **args)
    if kernel == "stdp":
        args["ARD1"] = False
        args["ARD2"] = True
        args["period"] = 360
        kernel = GPy.kern.StdPeriodic(**args)
    if kernel == "stdp2":
        args["ARD1"] = False
        args["ARD2"] = True
        args["period"] = 2
        kernel = GPy.kern.StdPeriodic(**args)
    # build gpy model
    data_X = data[:, 0:-1]
    Y_mean = np.mean(data[:, -1])
    data_Y = (data[:, -1] - Y_mean).reshape(-1, 1)
    model = GPy.models.GPRegression(data_X, data_Y, kernel=kernel)
    model.Gaussian_noise.variance = noise
    # model.optimize()
    # return a function
    return (model, Y_mean)


def generate_bossin(data, config):
    params = {}
    # userfn
    params['userfn'] = config.get("user_function", "user_function.py")
    # bounds
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    bounds = []
    r = config.get("space_pad", None)
    space_period = config.get("space_period", 0)
    for i in range(data_min.size - 1):
        if r is not None:
            _xmin = data_min[i] * (1+r) + data_max[i] * (-r)
            _xmax = data_min[i] * (-r) + data_max[i] * (1+r)
        else:
            _xmin, _xmax = - space_period / 2, space_period / 2
        bounds.append("%.3f %.3f" % (_xmin, _xmax))
    params['bounds'] = "; ".join(bounds)
    # yrange
    r = config.get("yrange", None)
    if r is not None:
        params['yrange'] = "%.3f %.3f" % tuple(r)
    else:
        r = config["energy_pad"]
        _ymin = data_min[-1] * (1+r) + data_max[-1] * (-r)
        _ymax = data_min[-1] * (-r) + data_max[-1] * (1+r)
        params['yrange'] = "%.3f %.3f" % (_ymin, _ymax)
    # noise
    params['noise'] = "%.3f" % config['noise']
    # initpts
    r = config.get("initpts", None)
    if r is not None:
        params['initpts'] = "%d" % r
    else:
        params['initpts'] = "%d" % data.shape[0]
    # iterpts
    params['iterpts'] = "%d" % config.get("iterpts", 0)
    # kernel
    params['kernel'] = config['kernel']
    # results
    results = ""
    for row in data:
        for x in row:
            results += "%27.18e" % x
        results += "\n"
    params['results'] = results
    # fill in
    content = None
    with open(config['template']) as f:
        content = f.read()
    for key, value in params.items():
        content = content.replace("__%s__" % key.upper(), value)
    return content


def check_bossout(key, boss_out):
    b = subprocess.check_output(
        ["grep", "-P", r"^\| " + key, "-A", "1", boss_out])
    text = b.decode().strip().split("\n")
    return [float(x) for x in text[-1].split()]
