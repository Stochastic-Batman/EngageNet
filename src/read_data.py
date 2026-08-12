import logging
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path



ROLES = ["expert", "novice"]

STREAM_FEATURES = [
    "audio.egemapsv2",
    "audio.w2vbert2_embeddings",
    "audio.xlm_roberta_embeddings",
    "openface2",
    "openpose",
    "clip",
    "dino",
    "imagebind",
    "swin",
    "videomae",
]

_SSI_DTYPE_MAP = {
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
    "SHORT": np.int16,
    "INT": np.int32,
    "CHAR": np.int8,
    "UCHAR": np.uint8,
    "BOOL": np.bool_,
    "LDOUBLE": np.float64,
}

def _make_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

log = _make_logger(f"EngageNet-Logger")



def read_stream(stream_path: str | Path) -> tuple[np.ndarray, float]:
    """
    Reads an SSI .stream/.stream~ pair.
    Returns (data, sample_rate) where data has shape (T, D).
    """
    stream_path = Path(stream_path)
    blob_path = Path(str(stream_path) + "~")

    root = ET.parse(stream_path).getroot()
    info = root.find("info")

    sr = float(info.attrib["sr"])
    dim = int(info.attrib["dim"])
    ftype = info.attrib["ftype"].upper()
    dtype = _SSI_DTYPE_MAP[info.attrib["type"].upper()]

    log.debug(f"stream {stream_path.name}: ftype={ftype} sr={sr} dim={dim} dtype={dtype.__name__}")

    if ftype == "ASCII":
        delim = info.attrib.get("delim", " ") or " "
        data = np.loadtxt(blob_path, delimiter=delim)
    else:
        chunks = root.findall("chunk")
        if chunks:
            parts = []
            with open(blob_path, "rb") as f:
                for chunk in chunks:
                    n = int(chunk.attrib["num"])
                    offset = int(chunk.attrib["byte"])
                    f.seek(offset)
                    parts.append(np.frombuffer(f.read(n * dim * np.dtype(dtype).itemsize), dtype=dtype))
            data = np.concatenate(parts)
        else:
            data = np.fromfile(blob_path, dtype=dtype)

    return data.reshape(-1, dim), sr


def read_engagement(path: str | Path) -> pd.Series:
    return pd.read_csv(path, sep=";", header=None, names=["score"])["score"]


def read_transcript(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", names=["start_time", "end_time", "content", "confidence"])


def read_scalar_annotation(path: str | Path) -> str:
    return Path(path).read_text().strip().split(";")[2]


def load_session(session_dir: str | Path, features: list[str] | None = None) -> dict:
    session_dir = Path(session_dir)
    log.info(f"loading session {session_dir}")
    session = {"path": str(session_dir), "expert": {}, "novice": {}}

    wanted = STREAM_FEATURES if features is None else [f for f in STREAM_FEATURES if f in features]

    for role in ROLES:
        r = session[role]

        for key, suffix in [("age", "age.annotation.csv"), ("gender", "gender.annotation.csv")]:
            p = session_dir / f"{role}.{suffix}"
            if p.exists():
                r[key] = read_scalar_annotation(p)
                log.debug(f"{role}.{key} = {r[key]}")

        p = session_dir / f"{role}.engagement.annotation.csv"
        if p.exists():
            r["engagement"] = read_engagement(p)
            log.debug(f"{role}.engagement: {len(r['engagement'])} frames @ 25Hz")

        p = session_dir / f"{role}.audio.transcript.annotation.csv"
        if p.exists():
            r["transcript"] = read_transcript(p)
            log.debug(f"{role}.transcript: {len(r['transcript'])} utterances")

        r["streams"] = {}
        for feat in wanted:
            p = session_dir / f"{role}.{feat}.stream"
            if p.exists():
                data, sr = read_stream(p)
                r["streams"][feat] = {"data": data, "sr": sr}
                log.info(f"{role}.{feat}: shape={data.shape} sr={sr}Hz dtype={data.dtype}")
            else:
                log.debug(f"{role}.{feat}: not found, skipping")

    p = session_dir / "language.annotation.csv"
    if p.exists():
        session["language"] = read_scalar_annotation(p)
        log.debug(f"language = {session['language']}")

    log.info(f"session {session_dir.name} loaded")
    return session


if __name__ == "__main__":
    import sys

    # Why 101? Because of the Dalmatians!
    default = Path(__file__).parent.parent / "data/NoXi+J/train/101"
    session_path = sys.argv[1] if len(sys.argv) > 1 else default
    session = load_session(session_path)

    for role in ROLES:
        t = session[role].get("transcript")
        if t is not None:
            log.info(f"{role} transcript preview:\n{t.head(3).to_string(index=False)}")
