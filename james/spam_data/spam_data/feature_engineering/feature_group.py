from abc import ABC, abstractmethod


class FeatureGroup(ABC):
    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def merge_data_to_base(self, base_df):
        pass

    def add_to(self, base_df):
        self.process_data()
        return self.merge_data_to_base(base_df)

    def __call__(self, base_df):
        self.process_data()
        return self.merge_data_to_base(base_df)
