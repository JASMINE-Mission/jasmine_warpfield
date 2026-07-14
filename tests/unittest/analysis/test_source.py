#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from pytest import approx, raises
import zodiax as zdx

from warpfield.analysis import SourceCatalog


def test_source_catalog():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])

    assert isinstance(source, zdx.Base)
    assert len(source) == 2
    assert source.take(jnp.array([1, 0]))[0] == approx([2.0, 1.0])
    assert source.take(jnp.array([1, 0]))[1] == approx([4.0, 3.0])
    assert len(jax.tree_util.tree_leaves(source)) == 2


def test_source_catalog_zodiax_update():
    source = SourceCatalog([1.0, 2.0], [3.0, 4.0])
    updated = source.set('ra', jnp.array([5.0, 6.0]))

    assert source.get('ra') == approx([1.0, 2.0])
    assert updated.get('ra') == approx([5.0, 6.0])


def test_source_catalog_shape_validation():
    with raises(ValueError, match='one-dimensional'):
        SourceCatalog([[1.0]], [2.0])
    with raises(ValueError, match='same shape'):
        SourceCatalog([1.0], [2.0, 3.0])
