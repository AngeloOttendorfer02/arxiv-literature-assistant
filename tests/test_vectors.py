import pytest

from scilitrag.rag.database import vector_literal
from scilitrag.rag.errors import RagUserError


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [1], [True, 0], ["1", "0"], None])
def test_invalid_vectors_fail_before_database_access(vector):
    with pytest.raises(RagUserError):
        vector_literal(vector, 2)


def test_valid_vector_is_serialized_for_a_bound_sql_parameter():
    assert vector_literal([1, 0], 2) == "[1.0,0.0]"
