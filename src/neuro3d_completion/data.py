"""Verified MNIST/Fashion-MNIST input files and immutable TRAIN/DEV/TEST exports."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import struct
import urllib.request
import numpy as np
from scipy.ndimage import zoom

from .evidence import digest_file, record

FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
HASHES = {
    "mnist": (
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
    "fashion": (
        "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
        "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
        "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
        "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5")}
BASE_URLS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"}


def read_idx(path):
    """Read the two supported unsigned-byte IDX formats with complete size checks."""
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 8:
        raise ValueError("truncated IDX header")
    magic, count = struct.unpack(">II", data[:8])
    if magic == 2049:
        if len(data) != 8 + count:
            raise ValueError("IDX label length mismatch")
        return np.frombuffer(data, np.uint8, offset=8).copy()
    if magic == 2051:
        if len(data) < 16:
            raise ValueError("truncated image dimensions")
        height, width = struct.unpack(">II", data[8:16])
        if height < 1 or width < 1 or len(data) != 16 + count*height*width:
            raise ValueError("IDX image length mismatch")
        return np.frombuffer(data, np.uint8, offset=16).reshape(count, height, width).copy()
    raise ValueError("unsupported IDX magic number")


def encode(images, size=64):
    """Resize positive amplitudes and normalize every nonzero example to unit energy."""
    x = np.asarray(images, dtype=float)
    if x.ndim != 3 or len(x) == 0 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("finite nonnegative image batches are required")
    if x.shape[1:] != (size, size):
        x = zoom(x, (1, size/x.shape[1], size/x.shape[2]), order=1, prefilter=False)
    energy = np.sqrt(np.sum(x*x, axis=(1, 2)))
    if np.any(energy == 0):
        raise ValueError("zero-energy images require a separately registered policy")
    return (x/energy[:, None, None]).astype(np.float64)


def save_split(path, images, labels, sample_ids):
    """Create a lossless split export without replacing an existing dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, images=images, labels=np.asarray(labels, dtype=np.int64),
                            sample_ids=np.asarray(sample_ids, dtype=np.int64))


def prepare(dataset, directory, output, offline=False):
    """Verify source bytes, then preserve the official split and a fixed DEV subset."""
    directory, output = Path(directory), Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, expected in zip(FILES, HASHES[dataset]):
        path = directory/filename
        if not path.exists():
            if offline:
                raise FileNotFoundError(path)
            temporary = path.with_suffix(path.suffix + ".part")
            if temporary.exists():
                raise FileExistsError(f"inspect the incomplete download: {temporary}")
            with urllib.request.urlopen(BASE_URLS[dataset]+filename, timeout=60) as response:
                with temporary.open("xb") as stream:
                    while block := response.read(1024*1024):
                        stream.write(block)
            if digest_file(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch in downloaded {filename}")
            temporary.rename(path)
        if digest_file(path) != expected:
            raise ValueError(f"SHA-256 mismatch in {filename}; no data has been admitted")
        paths.append(path)
    train_images, train_labels, test_images, test_labels = map(read_idx, paths)
    if (train_images.shape != (60000, 28, 28) or test_images.shape != (10000, 28, 28)
            or train_labels.shape != (60000,) or test_labels.shape != (10000,)
            or np.any(np.r_[train_labels, test_labels] > 9)):
        raise ValueError("unexpected official dataset dimensions or classes")
    # Preserve raw image bytes; resizing is identical for all models at training time.
    save_split(output/"train.npz", train_images[:50000], train_labels[:50000], np.arange(50000))
    save_split(output/"dev.npz", train_images[50000:], train_labels[50000:], np.arange(50000, 60000))
    save_split(output/"test.npz", test_images, test_labels, np.arange(10000))
    record(output/"manifest.json", "verified official dataset split", dict(dataset=dataset),
           dict(source_sha256={p.name: digest_file(p) for p in paths},
                split_sha256={name: digest_file(output/(name+".npz")) for name in ("train", "dev", "test")},
                train=50000, dev=10000, test=10000, writer_ids_available=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=HASHES)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(args.dataset, args.directory, args.output, args.offline)
    print(args.output/"manifest.json")


if __name__ == "__main__":
    main()
