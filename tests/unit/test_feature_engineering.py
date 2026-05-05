import pytest
from fraudguard.domain.models.call_event import CallEvent
from fraudguard.domain.models.feature_vector import FeatureVector


class TestFeatureEngineering:

    def setup_method(self) -> None:
        from fraudguard.ml.feature_engineering import PandasFeatureExtractor
        self.extractor = PandasFeatureExtractor()

    def test_extracts_six_features(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        assert result.values.shape == (1, 6)

    def test_returns_feature_vector_dataclass(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        assert isinstance(result, FeatureVector)
        assert len(result.feature_names) == 6

    def test_duration_normalized_between_0_and_1(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        idx = result.feature_names.index("duration_normalized")
        assert 0.0 <= result.values[0][idx] <= 1.0

    def test_duration_normalized_value(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        idx = result.feature_names.index("duration_normalized")
        assert result.values[0][idx] == pytest.approx(180 / 3600)

    def test_is_night_call_true_at_3am(self, night_call: CallEvent) -> None:
        result = self.extractor.extract(night_call)
        idx = result.feature_names.index("is_night_call")
        assert result.values[0][idx] == 1.0

    def test_is_night_call_false_at_2pm(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        idx = result.feature_names.index("is_night_call")
        assert result.values[0][idx] == 0.0

    def test_is_international_true_for_non_french_destination(
        self, fraud_call: CallEvent
    ) -> None:
        result = self.extractor.extract(fraud_call)
        idx = result.feature_names.index("is_international")
        assert result.values[0][idx] == 1.0

    def test_is_international_false_for_french_destination(
        self, normal_call: CallEvent
    ) -> None:
        result = self.extractor.extract(normal_call)
        idx = result.feature_names.index("is_international")
        assert result.values[0][idx] == 0.0

    def test_calls_last_hour_raw(self, fraud_call: CallEvent) -> None:
        result = self.extractor.extract(fraud_call)
        idx = result.feature_names.index("calls_last_hour")
        assert result.values[0][idx] == 45.0

    def test_sms_last_hour_raw(self, fraud_call: CallEvent) -> None:
        result = self.extractor.extract(fraud_call)
        idx = result.feature_names.index("sms_last_hour")
        assert result.values[0][idx] == 80.0

    def test_unique_dest_24h_raw(self, fraud_call: CallEvent) -> None:
        result = self.extractor.extract(fraud_call)
        idx = result.feature_names.index("unique_dest_24h")
        assert result.values[0][idx] == 50.0

    def test_feature_names_order(self, normal_call: CallEvent) -> None:
        result = self.extractor.extract(normal_call)
        assert result.feature_names == [
            "duration_normalized",
            "calls_last_hour",
            "is_night_call",
            "is_international",
            "sms_last_hour",
            "unique_dest_24h",
        ]
