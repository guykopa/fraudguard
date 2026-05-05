from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CallEvent:
    """Immutable raw telecom call event entering the inference pipeline."""

    msisdn: str
    duration_sec: float
    destination: str
    call_type: str
    hour_of_day: int
    calls_last_hour: int
    sms_last_hour: int
    unique_dest_24h: int
    timestamp: datetime
