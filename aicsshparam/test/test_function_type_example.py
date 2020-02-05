"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

from ..example import Example


def test_value_change():
    """An example of a simple function based test"""
    start_val = 5
    new_val = 20
    #
    example = Example(start_val)
    example.update_value(new_val)
    assert example.get_value() == new_val
    assert example.get_previous_value() == start_val
