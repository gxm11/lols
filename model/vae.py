# ---------------------------------------------------------
# variational autoencoder model
# ---------------------------------------------------------
# configs:
#   io_dims = input / latent / output
#   hidden layers
#   r_kld

import model.log as log
import torch
import subprocess
import argparse
import json
import os
import numpy as np
from torch import from_numpy, Tensor, nn, randn_like


class AutoEncoder(nn.Module):
    def __init__(self, io_dims, hidden_layers, r_kld, lr=0.01):
        super(AutoEncoder, self).__init__()
        self.input_dim, self.latent_dim, self.output_dim = io_dims
        self.hidden_layers = hidden_layers
        self.r_kld = r_kld

        self.build_network()
        self.setup_optimizer(lr=lr)

    def build_network(self):
        input_dim, output_dim, latent_dim = self.input_dim, self.output_dim, self.latent_dim
        hidden_layers = self.hidden_layers

        # encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_layers[0]))
        for i in range(len(hidden_layers) - 1):
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(
            nn.Linear(hidden_layers[-1], latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # vae encoder logvar
        encoder_layers.pop()
        encoder_layers.append(
            nn.Linear(hidden_layers[-1], latent_dim))
        self.encoder_logvar = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        reverse_layers = hidden_layers[::-1]
        decoder_layers.append(nn.Linear(latent_dim, reverse_layers[0]))
        for i in range(len(reverse_layers) - 1):
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(
                nn.Linear(reverse_layers[i], reverse_layers[i+1]))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(
            nn.Linear(reverse_layers[-1], output_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        mu = self.encoder(x)
        logvar = self.encoder_logvar(x)

        std = (logvar * 0.5).exp()
        eps = randn_like(std)
        z = mu + eps * std

        recon_x = self.decoder(z)

        return recon_x, mu, logvar

    def setup_optimizer(self, lr=0.01, weight_decay=0.0):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr, weight_decay=weight_decay
        )

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(
            torch.load(open(filename, "rb"))
        )


model = None


def train(model, epoches=1000):
    data = model.train_data
    optimizer = model.optimizer

    for _epoch in range(epoches):
        optimizer.zero_grad()
        recon_data, mu, logvar = model(data)
        loss, recon_loss = loss_function(recon_data, mu, logvar)
        loss.backward()
        optimizer.step()

    return loss.item(), recon_loss.item()


def setup(config, folder=None):
    log.info("setup variational auto-encoder")
    global model
    io_dims = config["io_dims"]
    hidden_layers = config.get("layers", [32, 32])
    r_kld = config.get("r_kld", 0.01)
    state_dict = config.get("state_dict", None)
    input_data = config.get("input_data", None)

    model = AutoEncoder(io_dims, hidden_layers, r_kld, lr=0.01)
    model.eval()

    if state_dict is not None:
        if os.path.exists(state_dict):
            model.load(state_dict)
            log.info(" - load state dict from <%s>" % state_dict)

    if input_data is not None:
        if os.path.exists(input_data):
            data = np.loadtxt(input_data, ndmin=2)
            data = data[:, 0:model.input_dim]
            model.train_data = from_numpy(data).float()
            t = (input_data, data.shape[0], data.shape[1])
            log.info(" - load input data from <%s>, shape: (%d, %d)" % t)

    if folder is not None:
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open("%s/config.json" % folder, "w") as f:
            json.dump(config, f, indent=4, sort_keys=True)


def run(workdir, execute, epoches, config=None):
    log.info("run variational auto-encoder at <%s> ..." % workdir)
    cmd = execute.split()
    if config is None:
        config = "%s/config.json" % workdir
    cmd += [__file__, config, "--workdir",
            workdir, "--epoches", "%d" % epoches]
    subprocess.call(cmd)

    b = subprocess.run(["grep", "Have a nice day", "vae.log"],
                       cwd=workdir, stdout=subprocess.DEVNULL)
    if b.returncode != 0:
        log.error(
            "Error happends on running vae, workdir: <%s>!" % workdir)
        return False
    else:
        return True


def high2low(data=None):
    global model
    if data is not None:
        data = from_numpy(data).float()
    else:
        data = model.train_data

    mu = model.encoder(data)
    logvar = model.encoder_logvar(data)

    mu = mu.detach().numpy()
    logvar = logvar.detach().numpy()

    return mu, logvar


def low2high(data):
    global model
    data = from_numpy(data).float()

    output = model.decoder(data)
    output = output.detach().numpy()
    output = (output + 1) % 2 - 1
    return output


def mae(data=None):
    if data is not None:
        data = from_numpy(data).float()
    else:
        data = model.train_data
    mu = model.encoder(data)
    output = model.decoder(mu)
    output = (output + 1).remainder(2) - 1
    # MAE
    a = data.detach().numpy()[:, 0:model.output_dim]
    b = output.detach().numpy()
    ae = (a - b + 1) % 2 - 1
    mae = np.mean(np.abs(ae))
    return mae


def mse(data=None):
    if data is not None:
        data = from_numpy(data).float()
    else:
        data = model.train_data
    mu = model.encoder(data)
    output = model.decoder(mu)
    output = (output + 1).remainder(2) - 1
    # MAE
    a = data.detach().numpy()[:, 0:model.output_dim]
    b = output.detach().numpy()
    ae = (a - b + 1) % 2 - 1
    mse = np.mean(ae * ae)
    return mse


if __name__ == "__main__":
    # training auto encoder
    parser = argparse.ArgumentParser(
        description="variational auto-encoder training"
    )

    parser.add_argument("config", type=str,
                        default="", help="Model configs")
    parser.add_argument("--workdir", "-W", type=str,
                        default=".", help="Work directory")
    parser.add_argument("--epoches", "-N", type=int,
                        default=100, help="Training Epoches")

    a = parser.parse_args()
    if not os.path.exists(a.workdir):
        os.mkdir(a.workdir)

    config = json.load(open(a.config))
    setup(config)
    loss = np.zeros((a.epoches, 2))
    min_step = int(a.epoches / np.e)
    min_loss = np.inf

    # -------------------------------------------
    # define loss function just before training
    # -------------------------------------------
    r_energy_w = config.get("r_energy_w", 0)
    w = 1
    if r_energy_w != 0:
        weights = (model.train_data[:, -1:] * r_energy_w).exp()
        w = weights / weights.mean()

    def loss_function(recon_x, mu, logvar):
        recon_x2 = model.train_data[:, 0:model.output_dim]
        diff = (recon_x - recon_x2 + 1).remainder(2) - 1
        mse = (diff.pow(2) * w).mean()
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return mse + model.r_kld * kld, mse

    model.train()

    log.info("start training...")
    for i in range(a.epoches):
        loss[i] = train(model, 1000)
        if i == min_step:
            if mae() > 0.25:
                raise log.error("vae underfitting.")
        if i > min_step and min_loss > loss[i, 0]:
            min_loss = loss[i, 0]
            model.save("%s/vae.pt" % a.workdir)
            mae_value = mae()
            mse_value = mse()
            t = (i, a.epoches, min_loss, mae_value, mae_value *
                 180, mse_value, mse_value ** 0.5 * 180)
            log.info(
                "(%d/%d) better model found. loss: %.8f, mae: %.6f (%.1f), mse: %.6f (%.1f)" % t)
    log.info("finish training.")

    model.load("%s/vae.pt" % a.workdir)
    model.eval()

    mu, logvar = high2low()
    recon = low2high(mu)
    np.savetxt("%s/mu.dat" % a.workdir, mu)
    np.savetxt("%s/logvar.dat" % a.workdir, logvar)
    np.savetxt("%s/recon.dat" % a.workdir, recon)
    np.savetxt("%s/vae_loss.dat" % a.workdir, loss)
    np.savetxt("%s/input_data.dat" % a.workdir,
               model.train_data.detach().numpy())
    # Done
    with open("%s/vae.log" % a.workdir, "w") as f:
        f.write("Auto Encoder is done. Have a nice day.")
