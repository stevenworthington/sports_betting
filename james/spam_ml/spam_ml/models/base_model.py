from abc import ABC, abstractmethod

from .. import serialization as ser


class Model(ABC):
    def __init__(self, name=None):
        self.model = None
        if name is not None:
            self.name = name
            self.path = ser.get_model_dir(name)
            self.load()
        else:
            self.name = ser.generate_unique_model_name()[1]
            self.path = ser.get_model_dir(self.name)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def save(self):
        ser.create_or_clean_dir(self.path)
        print(f"Saving model {self.name} to {self.path}")

    @abstractmethod
    def load(self):
        print(f"Attempting to load model {self.name} from {self.path}")

    # Adding these for posterity
    def get_model(self):
        return self.model

    def get_name(self):
        return self.name
