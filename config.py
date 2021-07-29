import json
import os

class Config(object):
    def __init__(self, config_filename: str = "config.json"):
        """
        instantiate a config object
        :param filepath: str
        absolute path to a json file
        """
        self.input_dict = self.from_json(config_filename)

        # raises error if wrong input is given
        self.check_dict()

    @staticmethod
    def get_config_path(config_filename: str):
        dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir, config_filename)
        return config_path

    def check_dict(self):
        """
        check if the input is a dict
        """
        if isinstance(self.input_dict, dict) is False:
            raise Exception("Error: the input to a config is not a dict")

    def from_json(self, config_filename: str):
        """
        reads a json with config dumps and returns a Config object.
        :param filepath: str
        absolute path to a json file

        :return: dict
        """
        filepath = self.get_config_path(config_filename)
        try:
            with open(filepath, "r") as fp:
                input_dict = json.load(fp)
        except:
            raise Exception('Could not read the file: {}'.format(filepath))
        return input_dict
