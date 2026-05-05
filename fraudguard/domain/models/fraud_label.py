from enum import Enum


class FraudLabel(Enum):
    FRAUD = "FRAUD"
    SUSPICIOUS = "SUSPICIOUS"
    LEGITIMATE = "LEGITIMATE"
