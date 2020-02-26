#!/usr/bin/env python
"""REST Translation server."""
from __future__ import print_function
import json
from onmt.utils.misc import check_model_config

from server_model import ServerModel, ServerModelError

class TranslationServer(object):
    def __init__(self):
        self.models = {}
        self.next_id = 0

    def start(self, config_file, debug):
        """Read the config file and pre-/load the models."""
        self.config_file = config_file
        with open(self.config_file) as f:
            self.confs = json.load(f)

        self.models_root = self.confs.get('models_root', './available_models')
        for i, conf in enumerate(self.confs["models"]):
            if "models" not in conf:
                if "model" in conf:
                    # backwards compatibility for confs
                    conf["models"] = [conf["model"]]
                else:
                    raise ValueError("""Incorrect config file: missing 'models'
                                        parameter for model #%s""" % i)
            check_model_config(conf, self.models_root)
            kwargs = {'timeout': conf.get('timeout', None),
                      'load': conf.get('load', None),
                      'preprocess_opt': conf.get('preprocess', None),
                      'tokenizer_opt': conf.get('tokenizer', None),
                      'postprocess_opt': conf.get('postprocess', None),
                      'on_timeout': conf.get('on_timeout', None),
                      'model_root': conf.get('model_root', self.models_root),
                      'debug': debug
                      }
            kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
            model_id = conf.get("id", None)
            opt = conf["opt"]
            opt["models"] = conf["models"]
            self.preload_model(opt, model_id=model_id, **kwargs)

    def clone_model(self, model_id, opt, timeout=-1):
        """Clone a model `model_id`.

        Different options may be passed. If `opt` is None, it will use the
        same set of options
        """
        if model_id in self.models:
            if opt is None:
                opt = self.models[model_id].user_opt
            opt["models"] = self.models[model_id].opt.models
            return self.load_model(opt, timeout)
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def load_model(self, opt, model_id=None, **model_kwargs):
        """Load a model given a set of options
        """
        model_id = self.preload_model(opt, model_id=model_id, **model_kwargs)
        load_time = self.models[model_id].load_time

        return model_id, load_time

    def preload_model(self, opt, model_id=None, **model_kwargs):
        """Preloading the model: updating internal datastructure

        It will effectively load the model if `load` is set
        """
        if model_id is not None:
            if model_id in self.models.keys():
                raise ValueError("Model ID %s already exists" % model_id)
        else:
            model_id = self.next_id
            while model_id in self.models.keys():
                model_id += 1
            self.next_id = model_id + 1
        print("Pre-loading model %s" % model_id)
        model = ServerModel(opt, model_id, **model_kwargs)
        self.models[model_id] = model

        return model_id

    def run(self, inputs, is_split=False):
        """Translate `inputs`

        We keep the same format as the Lua version i.e.
        ``[{"id": model_id, "src": "sequence to translate"},{ ...}]``

        We use inputs[0]["id"] as the model id
        """

        inputs_group_by_id = {}
        idx_group_by_id = {}

        for idx, input in enumerate(inputs):
            if "id" not in input:
                print("Error must set model id")
                raise ServerModelError("Error must set model id")

            model_id =  input.get("id")
            if model_id not in self.models:
                print("Error No such model '%s'" % str(model_id))
                raise ServerModelError("No such model '%s'" % str(model_id))

            if model_id not in inputs_group_by_id.keys():
                inputs_group_by_id[model_id] = [input]
                idx_group_by_id[model_id] = [idx]
            else:
                inputs_group_by_id[model_id].append(input)
                idx_group_by_id[model_id].append(idx)

        trans_tmp = []
        scores_tmp = []
        index_tmp = []

        for model_id, inputs_per_model_id in inputs_group_by_id.items():
            if model_id in self.models and self.models[model_id] is not None:
                trans = self.models[model_id].run(inputs_per_model_id, is_split)
                trans_tmp.extend(trans)
                scores_tmp.extend([None] * len(inputs_per_model_id))
            else:
                trans_tmp.extend([None]*len(inputs_per_model_id))
                scores_tmp.extend([None]*len(inputs_per_model_id))
            index_tmp.extend(idx_group_by_id[model_id])

        trans_total = []
        scores_total = []
        for idx in range(len(inputs)):
            trans_total.append(trans_tmp[index_tmp.index(idx)])
            scores_total.append(scores_tmp[index_tmp.index(idx)])

        return trans_total, scores_total

    def unload_model(self, model_id):
        """Manually unload a model.

        It will free the memory and cancel the timer
        """

        if model_id in self.models and self.models[model_id] is not None:
            self.models[model_id].unload()
        else:
            raise ServerModelError("No such model '%s'" % str(model_id))

    def list_models(self):
        """Return the list of available models
        """
        models = []
        for _, model in self.models.items():
            models += [model.to_dict()]
        return models


