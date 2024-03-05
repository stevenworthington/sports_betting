from abc import ABC, abstractmethod


class FeatureGroup(ABC):
    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def merge_data_to_base(self, base_df):
        pass
