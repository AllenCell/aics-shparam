"""
A simple example of a test using a class.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

from ..example import Example


class TestExample:

    START_VALUE = 5

    def test_initialize_value(self):
        example = Example(self.START_VALUE)
        assert example.get_value() == self.START_VALUE
