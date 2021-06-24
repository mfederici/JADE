def make_instance(class_name, modules, params=None, verbose=False):
    class_found = False

    if params is None:
        params = {}

    for module in modules:
        if hasattr(module, class_name):
            Class = getattr(module, class_name)
            class_found = True
            break

    if not class_found:
        raise Exception('No description for %s has been found in %s' % (
            class_name, str([module.__name__ for module in modules])))

    if verbose:
        print('Instantiating class %s from %s' %
              (class_name, module.__file__))

    instance = Class(**params)

    return instance


class InstanceManager:
    def __init__(self, descriptions, modules,  verbose=False, **extra_params):
        self.verbose = verbose
        if not isinstance(modules, list):
            modules = [modules]
        self.modules = modules

        self.loaded = dict()
        self.extra_params = extra_params

        self.descriptions = descriptions

    def _load(self, item):
        if not (item in self.descriptions):
            raise Exception('A description for %s has not been found in the description file' % item)

        class_name = self.descriptions[item]['class']
        if 'params' in self.descriptions[item]:
            params = self.descriptions[item]['params'].copy()
        else:
            params = dict()

        params.update(self.extra_params)

        self.loaded[item] = make_instance(class_name=class_name,
                                          modules=self.modules,
                                          params=params,
                                          verbose=self.verbose)

    def __getitem__(self, item):
        if not (item in self.loaded):
            self._load(item)
        return self.loaded[item]

    def get_config(self):
        return {k: self.descriptions[k] for k in self.loaded}
