import pytest
import smoke


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            case,
            id=case.name,
            marks=pytest.mark.xfail(reason=case.expected_fail_with_mocks, strict=False)
            if case.expected_fail_with_mocks
            else (),
        )
        for case in smoke.CASES
    ],
)
def test_smoke(case):
    assert smoke.run_case(case) == []
