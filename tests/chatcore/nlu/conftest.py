import pytest

from chatcore.nlu.entities import EntityExtractor


@pytest.fixture(scope="session")
def extractor(sample_bot):
    return EntityExtractor(sample_bot)
