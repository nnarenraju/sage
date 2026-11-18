"""GWTC population data releases.

Downloads a release's popsummary archive from Zenodo and extracts one
analysis's popsummary file.
"""

import hashlib
import tarfile
from pathlib import Path
from urllib.request import urlopen

# Zenodo records 16911563 (GWTC-4.0) and 20292639 (GWTC-5.0), CC BY 4.0.
RELEASES = {
    "gwtc4": {
        "url": "https://zenodo.org/api/records/16911563/files/analyses_BBH.tar/content",
        "archive": "analyses_BBH.tar",
        "md5": "ba554251bacbda3979206ba673c644f5",
    },
    "gwtc5": {
        "url": (
            "https://zenodo.org/api/records/20292639/files/"
            "popsummary_files.tar.gz/content"
        ),
        "archive": "popsummary_files.tar.gz",
        "md5": "d6ca03d47767d0e522f3c1647bf7e14e",
    },
}


def download_archive(release: str, directory: Path) -> Path:
    """Download a release's archive; reuse it if present.

    Parameters
    ----------
    release : str
        Key of ``RELEASES``.
    directory : Path
        Directory for the archive.

    Returns
    -------
    Path
        The archive.

    Raises
    ------
    ValueError
        If the release is unknown or the download does not match its MD5
        checksum.
    """
    if release not in RELEASES:
        raise ValueError(f"unknown release {release!r}; known: {sorted(RELEASES)}")
    source = RELEASES[release]
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / source["archive"]
    if archive.is_file():
        return archive

    partial = archive.with_name(f"{archive.name}.part")
    digest = hashlib.md5()
    with urlopen(source["url"], timeout=120) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 24):
            digest.update(chunk)
            handle.write(chunk)

    if digest.hexdigest() != source["md5"]:
        raise ValueError(f"{partial} has MD5 {digest.hexdigest()}, not {source['md5']}")
    partial.replace(archive)
    return archive


def extract_analysis(archive: Path, name: str) -> Path:
    """Extract one popsummary file beside the archive; reuse it if present.

    Parameters
    ----------
    archive : Path
        Output of ``download_archive``.
    name : str
        File name of the analysis inside the archive.

    Returns
    -------
    Path
        The extracted file.

    Raises
    ------
    ValueError
        If the archive holds no file of that name.
    """
    output = archive.parent / name
    if output.is_file():
        return output

    with tarfile.open(archive) as handle:
        members = {
            Path(member.name).name: member for member in handle if member.isfile()
        }
        if name not in members:
            raise ValueError(f"{archive} has no {name}; it holds {sorted(members)}")
        partial = output.with_name(f"{name}.part")
        partial.write_bytes(handle.extractfile(members[name]).read())
    partial.replace(output)
    return output
