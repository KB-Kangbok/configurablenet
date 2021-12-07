import os
import argparse
from filelock import FileLock

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, FIFOScheduler
from LRBench.framework.pytorch.utility import update_learning_rate
from LRBench.lr.piecewiseLR import piecewiseLR
from LRBench.lr.LR import LR

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

class ConfigurableNet:
  def __init__(self):
    self.dl = None
    self.nn = None
    self.dataloader = None
    self.config = None
    self.optim = None
    self.vision = None
  
  def set_searchspace(self, dl, net, dataloader, config, optim, vision):
    self.dl = dl
    self.net = net
    self.dataloader = dataloader
    self.config = config
    self.optim = optim
    self.vision = vision

  def data_loader(self, data_path):
    transform = self.vision.transforms.Compose(
        [self.vision.transforms.ToTensor(),
         self.vision.transforms.Normalize((0.1307, ), (0.3081, ))])
    
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser(f"{data_path}.lock")):
        train_loader = self.dataloader(
            self.vision.datasets.MNIST(
                data_path,
                train=True,
                download=True,
                transform=transform),
            batch_size=64,
            shuffle=True)
        test_loader = self.dataloader(
            self.vision.datasets.MNIST(
                data_path,
                train=False,
                download=True,
                transform=transform),
            batch_size=64,
            shuffle=True)
    return train_loader, test_loader

  def __train__(self, model, optimizer, train_loader, device):
    device = device
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = self.dl.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

  def test(self, model, data_loader, device):
    device = device
    model.eval()
    correct = 0
    total = 0
    with self.dl.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = self.dl.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

  def train(self, config):
    use_cuda = self.dl.cuda.is_available()
    device = self.dl.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = self.data_loader("~/data")
    model = self.net.to(device)
    # config = self.config

    # optimizer = self.optim.SGD(
    #     model.parameters(), lr=config["lr"], momentum=config["momentum"])
    optimizer = self.optim.Adadelta(model.parameters(), lr=config["lr"])
    lrbench_config =  {**config['lrBench'], **{'k0':config['lr']}} if config['lrBench']['lrPolicy'] == 'FIX' else config['lrBench']

    lrbenchLR = LR(lrbench_config)
    i = 1
    while True:
        self.__train__(model, optimizer, train_loader, device)
        acc = self.test(model, test_loader, device)
        update_learning_rate(optimizer, lrbenchLR.getLR(i-1))
        if config['lrBench']['lrPolicy'] == 'POLY':
            if i == config['lrBench']['l']+1:
                return

        i += 1
        # Set this to run Tune.
        tune.report(mean_accuracy=acc)

  def run(self):
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Enables GPU training")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.")
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using "
        "Ray Client.")
    args, _ = parser.parse_known_args()

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.shutdown()
        ray.init(num_cpus=2 if args.smoke_test else None)

    sched = FIFOScheduler()
    
    analysis = tune.run(
        lambda cfg: self.train(cfg),
        metric="mean_accuracy",
        mode="max",
        name="exp",
        scheduler=sched,
        stop={
            "training_iteration": self.config["stop_iteration"]
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": int(args.cuda)  # set this for GPUs
        },
        num_samples=1 if args.smoke_test else 1,
        config=self.config)
    
    print("Best config is:", analysis.best_config)


  