# ---------------------------------------------------------
# dft model (fhi-aims)
# ---------------------------------------------------------
import model.log as log
import model.convert as convert
import numpy as np
import subprocess
import os
import json


def setup(config, atoms, folder):
    log.info("setup fhi-aims...")
    workdir = folder
    if not os.path.exists(workdir):
        os.mkdir(workdir)

    with open("%s/config.json" % workdir, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    # geometry_in
    with open("%s/geometry.in" % workdir, "w") as f:
        e, pos = atoms
        contents = convert.generate_geometry_in(e, pos)
        f.write(contents)
    os.system("cp %s %s/control.in" % (config["template"], workdir))

    for key, value in config.items():
        if value is None:
            cmd = r"sed -i 's:\#*\(\s*\)%s:\#\1%s:' %s/control.in" % (
                key, key, workdir)
            os.system(cmd)
        else:
            cmd = r"sed -i 's:\#*\(\s*\)%s\(\s*\)\S*:\1%s\2%s:' %s/control.in" % (
                key, key, value, workdir)
            os.system(cmd)
    return workdir


def run(workdir, execute):
    log.info("run fhi-aims at <%s> ..." % workdir)
    cmd = execute.split()
    subprocess.call(cmd, cwd=workdir,
                    stdout=open("%s/aims.out" % workdir, "w"))

    b = subprocess.run(["grep", "Have a nice day", "aims.out"],
                       cwd=workdir, stdout=subprocess.DEVNULL)
    if b.returncode != 0:
        log.error("Error happends on running fhi-aims, workdir: <%s>!" % workdir)
        return False
    else:
        return True


def final_energy(workdir):
    try:
        b = subprocess.check_output(
            ["sed -n '/Final output/, /Before/p' aims.out | grep 'Total energy'"], shell=True, cwd=workdir)
        return float(b.decode().split()[-2])
    except:
        return None


def final_structure(workdir):
    try:
        b = subprocess.check_output(
            ["sed -n '/Final atomic structure:/, /---/p' aims.out"], shell=True, cwd=workdir)
        lines = b.decode().strip().split("\n")
        e, pos = {}, []
        index = 0
        for line in lines:
            if line.strip()[0:4] == "atom":
                _, x, y, z, element = line.split()
                index += 1
                e[index] = element
                pos.append(list(map(float, [x, y, z])))
        return e, np.array(pos)
    except:
        return None


def maximum_force(workdir):
    try:
        b = subprocess.check_output(
            ["grep", "Maximum force component", "aims.out"], cwd=workdir)
        text = b.decode().strip().split("\n")
        return float(text[-1].split()[4])
    except:
        return None


def is_converged(workdir):
    try:
        subprocess.check_output(
            ["grep", "Present geometry is converged", "aims.out"], cwd=workdir)
        return True
    except:
        return False


def is_finished(workdir):
    try:
        subprocess.check_output(
            ["grep", "Have a nice day", "aims.out"], cwd=workdir)
        return True
    except:
        return False


def all_structures(workdir):
    b = subprocess.check_output(
        ["sed -n '/[l|d] atomic structure:/, /---/p' aims.out"], shell=True, cwd=workdir)
    lines = b.decode().strip().split("\n")
    structures = []

    for line in lines:
        if 'atomic structure' in line:
            e, pos, index = {}, [], 0
        if '---' in line:
            structures.append((e, np.array(pos)))
        if line.strip()[0:4] == "atom":
            _, x, y, z, element = line.split()
            index += 1
            e[index] = element
            pos.append(list(map(float, [x, y, z])))

    return structures


def all_energies_and_forces(workdir):
    b = subprocess.check_output(
        ["sed -n '/Energy and forces in a compact form:/, /---/p' aims.out"], shell=True, cwd=workdir)
    lines = b.decode().strip().split("\n")
    energies = []
    forces = []
    flag_read_force = False
    for line in lines:
        if "Electronic free energy" in line:
            energy = float(line.split()[5])
        if "---" in line:
            energies.append(energy)
            forces.append(np.array(force))
            flag_read_force = False
        if "Total atomic forces" in line:
            flag_read_force = True
            force = []
            continue
        if flag_read_force:
            if len(line.split()) == 5:
                force.append(list(map(float, line.split()[2:5])))
    return energies, forces
