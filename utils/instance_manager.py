

class InstanceManager:
    def __init__(self, descriptions, modules,  verbose=False, **extra_params):
        self.verbose = verbose
        if not isinstance(modules, list):
            modules = [modules]
        self.modules = modules

        self.loaded = dict()
        self.extra_params = extra_params

        self.descriptions = descriptions

    def _make_instance(self, class_name, params):
        class_found = False

        for module in self.modules:
            if hasattr(module, class_name):
                Class = getattr(module, class_name)
                class_found = True
                break

        if not class_found:
            raise Exception('No description for %s has been found in %s' % (
                class_name, str([module.__name__ for module in self.modules])))

        if self.verbose:
            print('Instantiating class %s from %s' %
                  (class_name, module.__name__))

        instance = Class(**params)

        return instance

    def _load(self, item):
        if not (item in self.descriptions):
            raise Exception('A description for %s has not been found in the description file' % item)

        class_name = self.descriptions[item]['class']
        if 'params' in self.descriptions[item]:
            params = self.descriptions[item]['params'].copy()
        else:
            params = dict()

        params.update(self.extra_params)

        self.loaded[item] = self._make_instance(class_name, params)

    def __getitem__(self, item):
        if not (item in self.loaded):
            self._load(item)
        return self.loaded[item]

    def get_config(self):
        return {k: self.descriptions[k] for k in self.loaded}


class DatasetManager(InstanceManager):
    def _transform(self, instance, transforms):
        for transform in transforms:
            class_name = transform['class']
            if 'params' in transform:
                params = transform['params'].copy()
            else:
                params = dict()

            params['instance'] = instance
            instance = self._make_instance(class_name, params)
        return instance

    def _load(self, item):
        if not (item in self.descriptions):
            raise Exception('A description for %s has not been found in the description file' % item)

        descr = self.descriptions[item]
        if 'class' in descr:
            instance = self._make_instance(self.descriptions[item]['class'], self.descriptions[item]['params'])
        elif 'extend' in descr:
            instance = self.__getitem__(descr['extend']['base'])
        else:
            raise Exception('Either a class or the name of an instance to extend have to be specified')

        if 'transforms' in descr:
            instance = self._transform(instance, descr['transforms'])

        self.loaded[item] = instance



