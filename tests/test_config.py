import pytest

from pull_master.config import ConfigError, parse_repo


@pytest.mark.parametrize(
    "link, owner, name, issue",
    [
        ("https://github.com/acme/widgets", "acme", "widgets", None),
        ("https://github.com/acme/widgets.git", "acme", "widgets", None),
        ("https://github.com/acme/widgets/", "acme", "widgets", None),
        ("git@github.com:acme/widgets.git", "acme", "widgets", None),
        ("acme/widgets", "acme", "widgets", None),
        ("https://github.com/acme/widgets/issues/42", "acme", "widgets", 42),
        ("https://github.com/acme/widgets/issues/42#issuecomment-1", "acme", "widgets", 42),
        ("https://github.com/acme/widgets.js/tree/main", "acme", "widgets.js", None),
    ],
)
def test_parse_repo(link, owner, name, issue):
    ref = parse_repo(link)
    assert (ref.owner, ref.name, ref.issue) == (owner, name, issue)


@pytest.mark.parametrize("link", ["https://gitlab.com/a/b", "widgets", "https://github.com/acme"])
def test_parse_repo_rejects_other_links(link):
    with pytest.raises(ConfigError):
        parse_repo(link)
