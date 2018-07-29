# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""
from typing import List, Sequence
from flask import flash


def flash_errors(form, category='warning'):
    """Flash all errors for a form."""
    for field, errors in form.errors.items():
        for error in errors:
            flash('{0} - {1}'.format(getattr(form, field).label.text, error), category)


def split_integers(total: int, fractions: Sequence[float]) -> List[int]:
    '''
    Splits an integer into smaller ones as given by fractions.
    May not be exact.
    '''
    if sum(fractions) != 1.0:
        # If these fractions don't exactly add to 1...
        raise ValueError(f"Expected fractions {fractions} to add to 1.0, got {sum(fractions)}")

    splits = [round(total * f) for f in fractions]

    sum_splits = sum(splits)
    if sum_splits != total:
        # If rounding errors brought us just off the total...
        difference = total - sum_splits
        splits[0] += difference
        # This handles cases where the difference is both positive and negative
        # TODO: Is there a better way to distribute the difference?

    return tuple(splits)
