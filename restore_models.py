"""Reassemble the two split Allinone checkpoints and verify their SHA-256 hashes."""
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile


MODELS = {
    "Allinone/fitness_models.joblib":
        (6, "9af60bb724a0df98588d87fc841348d7c808e809336a32f2117d3addf5ea7d61"),
    "Allinone/Other_Exercises_Expanded_20261027/run_20261027_142919/fitness_models.joblib":
        (14, "986661fd1689a8cc8fc9851a987a366534a9c9c0b9574dd0960fc9d9836f2991"),
}


def copy_and_hash(source, digest, output=None):
    with source.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            if output is not None:
                output.write(block)


def restore(root, relative_path, count, expected_hash):
    target = root / relative_path
    digest = sha256()
    if target.exists():
        copy_and_hash(target, digest)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Existing file has a different SHA-256: {target}")
        return

    with NamedTemporaryFile(dir=target.parent, delete=False) as output:
        temporary = Path(output.name)
    try:
        with temporary.open("wb") as output:
            for index in range(1, count + 1):
                part = target.with_name(f"{target.name}.part{index:03d}")
                copy_and_hash(part, digest, output)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Parts failed SHA-256 verification: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    for path, (count, checksum) in MODELS.items():
        restore(root, path, count, checksum)
        print(f"Verified {path}")
