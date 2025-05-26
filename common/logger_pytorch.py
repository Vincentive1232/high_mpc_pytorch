
"""
A lightweight logging utility for PyTorch-based projects.
Adapted from OpenAI Baselines logger.
"""

import os
import sys
import json
import time
import datetime
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

class KVWriter:
    def writekvs(self, kvs):
        raise NotImplementedError

class SeqWriter:
    def writeseq(self, seq):
        raise NotImplementedError

class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, file_or_path):
        if isinstance(file_or_path, str):
            self.file = open(file_or_path, 'wt')
            self.own_file = True
        else:
            self.file = file_or_path
            self.own_file = False

    def writekvs(self, kvs):
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            valstr = f"{val:.3g}" if hasattr(val, "__float__") else str(val)
            key2str[key] = valstr
        if not key2str:
            print("WARNING: tried to write empty key-value dict")
            return
        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))
        dashes = "-" * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items()):
            lines.append(f"| {key:<{keywidth}} | {val:<{valwidth}} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    def writeseq(self, seq):
        self.file.write(" ".join(map(str, seq)) + "\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()

class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        new_keys = list(set(kvs.keys()) - set(self.keys))
        if new_keys:
            self.keys += sorted(new_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            self.file.write(",".join(self.keys) + "\n")
            for line in lines[1:]:
                self.file.write(line.strip() + self.sep * len(new_keys) + "\n")
        self.file.write(",".join(str(kvs.get(k, "")) for k in self.keys) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

class TensorBoardOutputFormat(KVWriter):
    def __init__(self, logdir):
        os.makedirs(logdir, exist_ok=True)
        self.writer = TorchSummaryWriter(log_dir=logdir)
        self.step = 0

    def writekvs(self, kvs):
        for k, v in kvs.items():
            try:
                self.writer.add_scalar(k, v, self.step)
            except Exception as e:
                print(f"[TensorBoardOutputFormat] Could not log {k}: {v} ({e})")
        self.writer.flush()
        self.step += 1

    def close(self):
        self.writer.close()

def make_output_format(format, ev_dir, suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(os.path.join(ev_dir, f"log{suffix}.txt"))
    elif format == "json":
        return JSONOutputFormat(os.path.join(ev_dir, f"progress{suffix}.json"))
    elif format == "csv":
        return CSVOutputFormat(os.path.join(ev_dir, f"progress{suffix}.csv"))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(os.path.join(ev_dir, f"tb{suffix}"))
    else:
        raise ValueError(f"Unknown format: {format}")

class Logger:
    CURRENT = None

    def __init__(self, dir, output_formats):
        self.name2val = defaultdict(float)
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    def logkv(self, key, val):
        self.name2val[key] = val

    def logkv_mean(self, key, val):
        oldval, count = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * count / (count + 1) + val / (count + 1)
        self.name2cnt[key] = count + 1

    def dumpkvs(self):
        for fmt in self.output_formats:
            fmt.writekvs(self.name2val)
        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            for fmt in self.output_formats:
                if isinstance(fmt, SeqWriter):
                    fmt.writeseq(map(str, args))

    def set_level(self, level): self.level = level
    def get_dir(self): return self.dir
    def close(self): [fmt.close() for fmt in self.output_formats]

def configure(dir=None, format_strs=None, suffix=""):
    if dir is None:
        dir = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("log-%Y%m%d-%H%M%S"))
    if format_strs is None:
        format_strs = ["stdout", "log", "json", "csv", "tensorboard"]
    output_formats = [make_output_format(fmt, dir, suffix) for fmt in format_strs]
    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)
    Logger.CURRENT.log(f"Logging to {dir}")
    return Logger.CURRENT

# API
def logkv(key, val): Logger.CURRENT.logkv(key, val)
def logkv_mean(key, val): Logger.CURRENT.logkv_mean(key, val)
def dumpkvs(): Logger.CURRENT.dumpkvs()
def log(*args, level=INFO): Logger.CURRENT.log(*args, level=level)
def set_level(level): Logger.CURRENT.set_level(level)
def get_dir(): return Logger.CURRENT.get_dir()
def close(): Logger.CURRENT.close()

# Demo
if __name__ == "__main__":
    configure()
    for i in range(5):
        logkv("loss", 0.1 * i)
        logkv("accuracy", 0.9 - 0.05 * i)
        dumpkvs()
