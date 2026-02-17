#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A base vectorizer for The Cannon.
"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["BaseVectorizer"]

import numpy as np


class BaseVectorizer(object):
    """
    A vectorizer class that models spectral fluxes and its derivatives.
    """

    def __init__(self, label_names, terms, **kwargs):
        self._terms = terms
        self._label_names = tuple(label_names)
        self.metadata = kwargs.get("metadata", {})
        return None


    # These can be over-written by sub-classes, but it is useful to have some
    # basic information if the sub-classes do not overwrite it.
    def __str__(self):
        return "<{module}.{name} object consisting of {K} labels and {D} terms>"\
            .format(module=self.__module__, name=type(self).__name__,
                D=len(self.terms), K=len(self.label_names))

    def __repr__(self):
        return "<{0}.{1} object at {2}>".format(
            self.__module__, type(self).__name__, hex(id(self)))


    # I/O (Serializable) functionality.
    def __getstate__(self):
        """ Return the state of the vectorizer. """
        return (type(self).__name__, dict(
            label_names=self.label_names, 
            terms=self.terms,
            metadata=self.metadata))


    def __setstate__(self, state):
        """ Set the state of the vectorizer. """
        # Expecting state in the form returned by __getstate__:
        # (class_name: str, kwds: dict)
        # Older pickles may have different shapes; try to be defensive.
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            _, kwds = state
        elif isinstance(state, dict):
            # Fallback: state is already the kwds dict
            kwds = state
        else:
            # As a last resort, try to interpret as only terms/labels present
            kwds = {"label_names": getattr(self, "_label_names", ()),
                    "terms": getattr(self, "_terms", ()),
                    "metadata": getattr(self, "metadata", {})}
        self._label_names = kwds["label_names"]
        self._terms = kwds["terms"]
        self.metadata = kwds["metadata"]


    @property
    def terms(self):
        """ Return the terms provided for this vectorizer. """
        return self._terms


    @property
    def label_names(self):
        """
        Return the label names that are used in this vectorizer.
        """
        return self._label_names


    def __call__(self, *args, **kwargs):
        """
        An alias to the get_label_vector method.
        """
        return self.get_label_vector(*args, **kwargs)


    def get_label_vector(self, labels, *args, **kwargs):
        """
        Return the label vector based on the labels provided.

        :param labels:
            The values of the labels. These should match the length and order of
            the `label_names` attribute.
        """
        raise NotImplementedError("the get_label_vector method "
                                  "must be specified by the sub-classes")


    def get_label_vector_derivative(self, labels, *args, **kwargs):
        """
        Return the derivative of the label vector with respect to the given
        label.

        :param labels:
            The values of the labels to calculate the label vector for.
        """
        raise NotImplementedError("the get_label_vector_derivative method "
                                  "must be specified by the sub-classes")