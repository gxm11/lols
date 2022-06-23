# structure convertion model
# load and convert structures into different formats

import numpy as np
import model.log as log


def load_gzmat(fn):
    e, s, data = {}, {}, {}
    flag_structure = False
    flag_variable = False
    with open(fn) as f:
        for line in f.readlines():
            if line.split() == ["0", "1"]:
                flag_structure = True
                flag_variable = False
                index = 0
                continue
            if line[0:10] == "Variables:":
                flag_structure = False
                flag_variable = True
                continue

            if flag_structure:
                index += 1
                x = line.split()
                e[index] = x[0]
                s[index] = [index] + [int(i) for i in x[1::2]]

            if flag_variable:
                x = line.split("=")
                if len(x) == 2:
                    data[x[0]] = float(x[1])
    return e, s, data


def load_xyz(fn):
    es = np.loadtxt(fn, dtype=np.str, skiprows=2, usecols=0)
    pos = np.loadtxt(fn, skiprows=2, usecols=[1, 2, 3])
    e = {}
    for i in range(es.size):
        e[i + 1] = es[i]
    return e, pos


def load_geometryin(fn):
    e = {}
    pos = []
    i = 0
    with open(fn) as f:
        for line in f.readlines():
            if line[0:4] == "atom":
                x = line.split()
                e[i + 1] = x[4]
                pos.append(list(map(float, x[1:4])))
                i += 1
    pos = np.array(pos)
    return e, pos


def generate_position(s, data):
    pos = []
    for i in range(len(s.keys())):
        index = i + 1
        if index == 1:
            pos.append([0.0, 0.0, 0.0])
        if index == 2:
            pos.append([data["r2"], 0.0, 0.0])
        if index == 3:
            if s[index] == [3, 2, 1]:
                x = data["r2"] - data["r3"] * np.cos(data["a3"] * np.pi / 180)
            else:
                x = data["r3"] * np.cos(data["a3"] * np.pi / 180)

            z = data["r3"] * np.sin(data["a3"] * np.pi / 180)
            pos.append([x, 0.0, z])
        if index > 3:
            pos_1 = np.array(pos[s[index][1] - 1])
            pos_2 = np.array(pos[s[index][2] - 1])
            pos_3 = np.array(pos[s[index][3] - 1])
            e1 = pos_2 - pos_1
            e2 = np.cross(pos_2 - pos_1, pos_2 - pos_3)
            e1 = e1 / np.linalg.norm(e1)
            e2 = e2 / np.linalg.norm(e2)
            e3 = np.cross(e1, e2)

            a = data["a%d" % index] * np.pi / 180
            b = data["d%d" % index] * np.pi / 180

            e0 = e1 * np.cos(a) + np.sin(a) * (e2 * np.sin(b) + e3 * np.cos(b))

            pos.append(data["r%d" % index] * e0 + pos_1)
    return np.array(pos)


def generate_variable(s, pos):
    data = {}
    for i in range(len(s.keys())):
        index = i + 1
        if index >= 2:
            pos_0 = np.array(pos[s[index][0] - 1])
            pos_1 = np.array(pos[s[index][1] - 1])
            data["r%d" % index] = np.linalg.norm(pos_0 - pos_1)
        if index >= 3:
            pos_2 = np.array(pos[s[index][2] - 1])
            v1 = pos_1 - pos_0
            v2 = pos_1 - pos_2
            v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
            data["a%d" % index] = np.arccos(np.dot(v1, v2)) * 180 / np.pi % 360
        if index >= 4:
            pos_3 = np.array(pos[s[index][3] - 1])
            v3 = pos_3 - pos_2

            u1 = np.cross(v1, v2)
            u2 = np.cross(v2, v3)
            u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)

            sign = -np.sign(np.dot(v1, np.cross(v2, v3)))
            data["d%d" % index] = np.arccos(
                np.dot(u1, u2)) * 180 / np.pi * sign % 360
    return data


def render_structure(s, e):
    text = []
    for i in range(len(s.keys())):
        index = i + 1
        if index == 1:
            text.append(e[index])
        if index == 2:
            text.append("%s %d r2" % (e[index], s[index][1]))
        if index == 3:
            text.append("%s %d r3 %d a3" %
                        (e[index], s[index][1], s[index][2]))
        if index > 3:
            t = "%s %d r__index__ %d a__index__ %d d__index__"
            t = t % (e[index], *s[index][1:])
            t = t.replace("__index__", "%d" % index)
            text.append(t)
    return "\n".join(text)


def generate_data(data, **D):
    d = data.copy()

    flag_retry = True
    while flag_retry:
        D.update(d)
        flag_retry = False
        for key in d.keys():
            if type(d[key]) == str:
                try:
                    d[key] = eval(d[key].format(**D))
                except:
                    flag_retry = True
                    continue
    return d


def generate_geometry_in(e, pos):
    text = []
    for i in e.keys():
        a = list(pos[i - 1, :])
        a.append(e[i])
        text.append("atom  %12f  %12f  %12f  %s" % tuple(a))

    return "\n".join(text)


def generate_xyz(e, pos, comment="structure"):
    text = [str(len(e.keys())), comment]
    for i in e.keys():
        a = [e[i]]
        a += list(pos[i - 1, :])
        text.append("%3s  %12f  %12f  %12f" % tuple(a))

    return "\n".join(text)


def auto_refine(e, pos, template, radius_error=0.2, angle_error=30, dihe_error=90):
    args = []
    best = [180, None]
    _dfs_refine(args, best, e, pos, template,
                radius_error, angle_error, dihe_error)
    if best[1] is None:
        _dfs_refine(args, best, e, pos, template,
                    radius_error, angle_error, 180)
    # 20220106 fix: if cannot found any result, keep pos unchanged
    if best[1] is None:
        return template["elements"], pos

    _pos = pos[best[1]]
    _pos = _pos - _pos[0]
    return template["elements"], _pos


def _dfs_refine(args, best, e, pos, template, radius_error, angle_error, dihe_error):
    elements = template["elements"]

    n = len(args)
    if n == len(elements):
        error = _final_error(pos[args], template)
        if error < best[0]:
            best[0] = error
            best[1] = args
        return

    for i, _e in e.items():
        if i - 1 in args:
            continue
        if _e != elements[n + 1]:
            continue

        _args = args + [i - 1]
        if not _check_pos(pos[_args], template, radius_error, angle_error, dihe_error):
            continue

        _dfs_refine(_args, best, e, pos, template,
                    radius_error, angle_error, dihe_error)


def _final_error(pos, template):
    data = template["data"]
    data_from_pos = generate_variable(template["structure"], pos)
    data_format = {}
    errors = []

    for k, v in data.items():
        if type(v) == str:
            if v[0:2] == "{D":
                data_format[k] = data_from_pos[k]
        else:
            data_format[k] = v

    for k, v in data.items():
        d = data_from_pos[k]
        if type(v) == str:
            if v[0:2] == "{D":
                continue
            _d = eval(v.format(**data_format))
            error = np.abs((_d - d + 180) % 360 - 180)
            errors.append(error)

    return np.mean(errors)


def _check_pos(_pos, template, radius_error, angle_error, dihe_error):
    structure = template["structure"]
    data = template["data"]
    # only check the final one
    k = _pos.shape[0]
    v = [x - 1 for x in structure[k]]
    if len(v) >= 2:
        r = data["r%d" % k]
        _r = np.linalg.norm(_pos[v[0]] - _pos[v[1]])
        if np.abs(r - _r) > radius_error:
            return False
    if len(v) >= 3:
        a = data["a%d" % k]
        v1 = _pos[v[1]] - _pos[v[0]]
        v2 = _pos[v[1]] - _pos[v[2]]
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        _a = np.arccos(np.dot(v1, v2)) * 180 / np.pi % 360
        if np.abs((_a - a + 180) % 360 - 180) > angle_error:
            return False
    if len(v) >= 4 and dihe_error < 180:
        d = data["d%d" % k]
        if type(d) is float:
            v3 = _pos[v[3]] - _pos[v[2]]
            u1 = np.cross(v1, v2)
            u2 = np.cross(v2, v3)
            u1, u2 = u1 / np.linalg.norm(u1), u2 / np.linalg.norm(u2)

            sign = -np.sign(np.dot(v1, np.cross(v2, v3)))
            if sign == 0:
                sign = 1
            _d = np.arccos(np.dot(u1, u2)) * 180 / np.pi * sign % 360
            if np.abs((_d - d + 180) % 360 - 180) > dihe_error:
                return False
    return True
