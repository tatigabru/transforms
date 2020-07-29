import random
from warnings import warn
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable, Optional


__all__ = ["BasicTransform", "DualTransform", "NoOp"]


class BasicTransform(ABC):
    """
    An abstraction knows how to apply transform.
    Provides the summary of variables used in this abstract class.
    Contains information about the methods for transformations used.    
    """
    def __init__(self, 
    p: float=0.5):
        """
        Args:
            p (float): probability of applying transform, between 0 and 1
            deterministic (bool): if True fixes all seeds for reproducibility. Default = False
        """
        self.p = p
        self.additional_targets = {}
        self.deterministic = False 
        self.seed: int       
        self.params = {}
    
    @abstractmethod
    def fix_seed(self, seed: int = 1234):
        """
        Fix initial seed, used to setup a seed for all randoms used
        for transforms
        Used for experiment reproducibility.
        """
        pass

    def __call__(self, samples, sample_rate):
        if self.deterministic:
            fix_seed(self, seed)
        if self.parameters["should_apply"] and len(samples) > 0:
            return self.apply(samples, sample_rate)
        return samples


    @abstractmethod
    def apply(self, signal):
        raise NotImplementedError


    @abstractmethod
    def get_parameters(self) -> Dict:
        """Return the parameters for serialization"""
        pass

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    

        

    def __call__(self, force_apply=False, **kwargs):
        if (random.random() < self.p):
            params = self.get_params()

            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname() + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params, force_apply=False, **kwargs):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {k: kwargs[k] for k in self.target_dependence.get(key, [])}
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag, save_key="replay"):
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(name=self.__class__.__name__, args=format_args(state))

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        params.update({"cols": kwargs["image"].shape[1], "rows": kwargs["image"].shape[0]})
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'target_image': 'image'}
        ex: {'obj1_mask': 'mask', 'obj2_mask': 'mask'}
        by the way you must have at least one object with key 'image'
        Args:
            additional_targets (dict): keys - new target name, values - old target name. ex: {'image2': 'image'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class " + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self):
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self):
        d = self._to_dict()
        d["id"] = id(self)
        return d


class DualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]

    def apply_to_keypoints(self, keypoints, **params):
        return [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

    def apply_to_masks(self, masks, **params):
        return [self.apply_to_mask(mask, **params) for mask in masks]


class ImageOnlyTransform(BasicTransform):
    """Transform applied to image only."""

    @property
    def targets(self):
        return {"image": self.apply}


class NoOp(DualTransform):
    """Does nothing"""

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, img, **params):
        return img

    def get_transform_init_args_names(self):
        return ()