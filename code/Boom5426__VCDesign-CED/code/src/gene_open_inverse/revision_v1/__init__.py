"""Corrections made during revision, kept separate from the submitted modules.

``grouped_fit`` replaces the ridge penalty selection of ``four_context_v1.predictors``:
folds are cut over gene identities rather than row indices, and the response basis is
refit inside each fold. ``four_context_v1.predictors.fit`` uses it by default and takes
``selection="frozen"`` to reproduce the submitted selection exactly. On the published
pools the selected penalty is unchanged.
"""
