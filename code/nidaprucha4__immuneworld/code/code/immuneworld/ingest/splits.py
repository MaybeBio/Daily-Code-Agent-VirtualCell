from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Partition:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def perturbation_identity_split(identities: np.ndarray, seed: int) -> Partition:
    unique = np.unique(identities)
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique)
    train_end = int(len(shuffled) * 0.8)
    validation_end = int(len(shuffled) * 0.9)
    train_ids = set(shuffled[:train_end].tolist())
    validation_ids = set(shuffled[train_end:validation_end].tolist())
    test_ids = set(shuffled[validation_end:].tolist())
    train = np.flatnonzero(np.asarray([value in train_ids for value in identities]))
    validation = np.flatnonzero(np.asarray([value in validation_ids for value in identities]))
    test = np.flatnonzero(np.asarray([value in test_ids for value in identities]))
    return Partition(train, validation, test)


def leave_one_cohort_out(cohorts: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    for cohort in np.unique(cohorts):
        test = np.flatnonzero(cohorts == cohort)
        train = np.flatnonzero(cohorts != cohort)
        yield str(cohort), train, test


def leave_one_cancer_out(cancers: np.ndarray) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    for cancer in np.unique(cancers):
        test = np.flatnonzero(cancers == cancer)
        train = np.flatnonzero(cancers != cancer)
        yield str(cancer), train, test


def assert_disjoint_patients(train: np.ndarray, test: np.ndarray, patients: np.ndarray) -> None:
    overlap = set(patients[train].tolist()) & set(patients[test].tolist())
    if overlap:
        raise ValueError("patient identifiers overlap between partitions")
