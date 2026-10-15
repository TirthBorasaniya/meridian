"""Paper metadata records shared by the arXiv client, the manifest, and the chunker."""

from dataclasses import dataclass, field


@dataclass
class PaperMeta:
    """Metadata of one versioned arXiv paper."""

    arxiv_id: str
    version: int
    title: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str] = field(default_factory=list)
    abstract: str = ""
    comment: str = ""
    pdf_url: str = ""
    o_withdrawn: bool = False

    @property
    def versioned_id(self) -> str:
        """Return `{arxiv_id}v{version}`."""
        return f"{self.arxiv_id}v{self.version}"

    @property
    def abs_url(self) -> str:
        """Return the versioned arXiv abstract page URL."""
        return f"https://arxiv.org/abs/{self.versioned_id}"


def split_versioned_id(versioned_id: str) -> tuple[str, int]:
    """
    Split `2201.11903v6` into (`2201.11903`, 6).

    Parameters
    ----------
    versioned_id : str
        An arXiv identifier with a `v<n>` suffix.

    Returns
    -------
    parts : tuple of (str, int)
        Base identifier and version number.
    """
    base_id, separator, version_text = versioned_id.rpartition("v")
    if not separator or not base_id or not version_text.isdigit():
        raise ValueError(f"not a versioned arXiv id: {versioned_id!r}")
    return base_id, int(version_text)
