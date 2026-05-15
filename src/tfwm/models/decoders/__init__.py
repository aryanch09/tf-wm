"""Prediction head decoders for TF-WM."""

from __future__ import annotations

from tfwm.models.decoders.affordance_head import AffordanceHead
from tfwm.models.decoders.contact_event_head import ContactEventHead
from tfwm.models.decoders.object_state_head import ObjectStateHead
from tfwm.models.decoders.tactile_decoder import TactileDecoder

__all__ = ["TactileDecoder", "ContactEventHead", "AffordanceHead", "ObjectStateHead"]
