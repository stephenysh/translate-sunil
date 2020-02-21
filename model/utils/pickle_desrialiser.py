import pickle

class CustomUnpickler(pickle.Unpickler):
    module_name = 'model.utils'

    def find_class(self, module, name):
        if module == "__main__":
            module = self.module_name
        return super().find_class(module, name)
