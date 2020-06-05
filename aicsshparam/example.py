import logging

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Example:
    def __init__(self, init_value: int = 10):
        self.current_value = init_value
        self.old_value = 0

    def update_value(self, new_value: int):
        """
        Save old value and set new value
        :param new_value: The new value to assign to the object
        :return: The old value
        """
        self.old_value = self.current_value
        self.current_value = new_value
        log.info(f"Updating value from {self.old_value} to {self.current_value}")
        return self.old_value

    def get_value(self) -> int:
        """
        :return: The current value of the object
        """
        return self.current_value

    def get_previous_value(self) -> int:
        """
        :return: The previous value of the object before it was updated
        """
        return self.old_value
