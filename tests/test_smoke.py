import pytest
import smoke


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case.name,
            marks=pytest.mark.xfail(reason=smoke.expected_fail(case), strict=False)
            if smoke.expected_fail(case)
            else (),
        )
        for case in smoke.CASES
    ],
)
def test_smoke(case):
    assert smoke.run_case(case) == []
