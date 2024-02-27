from abc import ABC, abstractmethod

class FeatureGroup(ABC):
    @abstractmethod
    def load_data_from_cache(self):
        pass

    @abstractmethod
    def merge_data_to_base(self):
        pass
