from logging import getLogger


class BaseLoader:
    def __init__(self, path):
        super(BaseLoader, self).__init__()
        self.logger = getLogger(__name__)
        self.path = path

    # def get_schema(self) -> Schema:
    #     raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
