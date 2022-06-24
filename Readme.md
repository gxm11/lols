# Low-energy Latent Space Search Workflow
![](./lols.jpeg "lols-toc-figure")

## Tasks
Tasks in the LOLS workflow.
- main
  * data_generation
    - train_vae
    - sample
    - dft_energy
  * energy_model
    - gp_model
    - gp_minimum
  * relaxation
    - dft_relax

Read our publication [Molecular Conformer Search with Low-Energy Latent Space](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00290) for details.

## Run
1. Switch to the python env which have **pytorch** and [BOSS](https://pypi.org/project/aalto-boss/) installed.
 - pytorch version: `1.3.1`
 - BOSS version: `0.9.17`
2. Create `config.json`.
 - Check the template files in `lib/config`.
 - If there's no `config.json`, the `config-test.json` will be used.
3. Run `python main.py`.

## Restart Task
Set task's state = 0 or 1 to restart that task:
```bash
sqlite3 lols.db 'update task set state = {new_state} where id = {task_id}'
```

- If the state set to 0, the task will refresh its config before running.

## Database
table task:
| column | information |
|:------:|:-----------:|
| id | task id, the primary key |
| state | 0: not start, 1: start, 2: finish |
| iteration | number of iteration |
| key | sub index |
| name | name of task, related to task/{name}.py |
| config | config of task, will save in config.json |

## Folders and Files
| folder | information |
|:------:|:-----------:|
| lib | read-only files for materials and software configure |
| model | Encapsulation of VAE model and other physical models. |
| task | LOLS tasks |
| work | Isolated folders for all tasks. |

| file | information |
|:----:|:-----------:|
| config.json | Where configurations for all tasks are saved. |
| lols.db | data base, where only the task information are saved. |
| database.py | python modules for operating `lols.db` |
| main.py | entries for the program |

## Hack BOSS
We need only the Gaussian Process fitting `hyper_parameters`, not the `x_next` and `global_minimum`. So we hacked **BOSS** library to add the `s` model, by replaced the source code with `lib/boss_modify/*`.

Since the current boss version is `1.5` (Checked at 2022-06-24), it might provide the options to skip finding the `x_next` or `global_minimum`.

Use it at your own risk.

You can also replace `lib/gp_model.py` to use the [GPy](https://pypi.org/project/GPy/) library directly or other Gaussian Process library like [gpytorch](https://gpytorch.ai/).