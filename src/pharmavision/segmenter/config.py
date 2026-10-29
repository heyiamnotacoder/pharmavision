"""The backend is chosen by config, so another backend (self-hosted SAM 3.1 on open weights)
can be added without touching the stages: add its settings to the union and a case below."""

from . import Segmenter
from .meta_api import MetaApiSegmenter, MetaApiSettings

SegmenterSettings = MetaApiSettings
"""One backend so far; becomes a union discriminated on `backend` when a second arrives."""


def make_segmenter(settings: SegmenterSettings) -> Segmenter:
    match settings:
        case MetaApiSettings():
            return MetaApiSegmenter(settings)
