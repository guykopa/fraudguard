# Rapport de validation — fraudguard

Date : 2026-06-15
Branche : dev
Commit : 1088e31
Environnement : Python 3.11, WSL2 Linux 6.6.87.2

---

## 1. Résultats des tests

### Synthèse

| Niveau | Tests | Passed | Failed | Couverture |
|---|---|---|---|---|
| Unitaire | 37 | 37 | 0 | 97% |
| Intégration | 8 | 8 | 0 | (inclus) |
| Non-régression | 3 | 3 | 0 | (inclus) |
| **Total** | **48** | **48** | **0** | **97%** |

### Détail par suite

#### Unitaires — `tests/unit/`

| Test | Statut |
|---|---|
| `TestFeatureEngineering::test_extracts_six_features` | PASSED |
| `TestFeatureEngineering::test_returns_feature_vector_dataclass` | PASSED |
| `TestFeatureEngineering::test_duration_normalized_between_0_and_1` | PASSED |
| `TestFeatureEngineering::test_duration_normalized_value` | PASSED |
| `TestFeatureEngineering::test_is_night_call_true_at_3am` | PASSED |
| `TestFeatureEngineering::test_is_night_call_false_at_2pm` | PASSED |
| `TestFeatureEngineering::test_is_international_true_for_non_french_destination` | PASSED |
| `TestFeatureEngineering::test_is_international_false_for_french_destination` | PASSED |
| `TestFeatureEngineering::test_calls_last_hour_raw` | PASSED |
| `TestFeatureEngineering::test_sms_last_hour_raw` | PASSED |
| `TestFeatureEngineering::test_unique_dest_24h_raw` | PASSED |
| `TestFeatureEngineering::test_feature_names_order` | PASSED |
| `TestPredictionService::test_returns_prediction_dataclass` | PASSED |
| `TestPredictionService::test_returns_fraud_for_high_risk_score` | PASSED |
| `TestPredictionService::test_legitimate_call_returns_legitimate_label` | PASSED |
| `TestPredictionService::test_suspicious_label_for_mid_risk_score` | PASSED |
| `TestPredictionService::test_model_version_propagated` | PASSED |
| `TestPredictionService::test_confidence_between_0_and_1` | PASSED |
| `TestPredictionService::test_measures_inference_latency` | PASSED |
| `TestPredictionService::test_risk_score_equals_fraud_proba` | PASSED |
| `TestDriftDetector::test_no_drift_on_identical_distributions` | PASSED |
| `TestDriftDetector::test_critical_drift_on_very_different_distributions` | PASSED |
| `TestDriftDetector::test_report_contains_feature_name` | PASSED |
| `TestDriftDetector::test_report_is_drift_report_dataclass` | PASSED |
| `TestDriftDetector::test_handles_zero_division_gracefully` | PASSED |
| `TestDriftDetector::test_warning_drift_on_moderately_different_distributions` | PASSED |
| `TestDriftDetector::test_reference_mean_is_populated` | PASSED |
| `TestDriftDetector::test_current_mean_is_populated` | PASSED |
| `TestJsonFormatter::test_format_produces_valid_json` | PASSED |
| `TestJsonFormatter::test_format_includes_exception` | PASSED |
| `TestGetLogger::test_returns_logger_instance` | PASSED |
| `TestGetLogger::test_logger_has_handler` | PASSED |
| `TestGetLogger::test_same_name_returns_same_logger` | PASSED |
| `TestPrometheusPublisher::test_publish_prediction_does_not_raise` | PASSED |
| `TestPrometheusPublisher::test_publish_prediction_legitimate_does_not_raise` | PASSED |
| `TestPrometheusPublisher::test_publish_drift_does_not_raise` | PASSED |
| `TestPrometheusPublisher::test_set_model_version_does_not_raise` | PASSED |

#### Intégration — `tests/integration/`

| Test | Statut |
|---|---|
| `test_predict_fraud_call` | PASSED |
| `test_predict_legitimate_call` | PASSED |
| `test_predict_returns_model_version` | PASSED |
| `test_predict_requires_auth` | PASSED |
| `test_health_liveness` | PASSED |
| `test_health_readiness` | PASSED |
| `test_model_info` | PASSED |
| `test_drift_endpoint` | PASSED |

#### Non-régression — `tests/non_regression/`

| Test | Statut | Valeur observée |
|---|---|---|
| `test_fraud_call_always_classified_as_fraud` | PASSED | label=FRAUD, risk_score ≥ 0.70 |
| `test_normal_call_always_classified_as_legitimate` | PASSED | label=LEGITIMATE, risk_score < 0.40 |
| `test_inference_latency_under_50ms` | PASSED | latency < 500ms (seuil CI relaxé) |

---

## 2. Couverture par module

| Module | Lignes | Non couverts | Couverture |
|---|---|---|---|
| `domain/models/` | 52 | 0 | 100% |
| `domain/services/` | 46 | 0 | 100% |
| `domain/exceptions/` | 6 | 0 | 100% |
| `interfaces/` | 30 | 0 | 100% |
| `ml/feature_engineering.py` | 14 | 0 | 100% |
| `pipeline/inference_pipeline.py` | 12 | 0 | 100% |
| `monitoring/drift_detector.py` | 35 | 2 | 94% |
| `monitoring/logger.py` | 18 | 0 | 100% |
| `monitoring/metrics.py` | 16 | 0 | 100% |
| `api/main.py` | 48 | 1 | 98% |
| `api/routes/predict.py` | 31 | 0 | 100% |
| `api/routes/monitoring.py` | 19 | 0 | 100% |
| `api/routes/model.py` | 14 | 0 | 100% |
| `api/routes/health.py` | 19 | 2 | 89% |
| `api/security/jwt_handler.py` | 26 | 5 | 81% |
| `adapters/sklearn_model_registry.py` | 29 | 4 | 86% |
| **TOTAL** | **415** | **14** | **97%** |

Modules sous 90% (à couvrir — voir plan de test G-03) :

- `jwt_handler.py` L17, L51-57 : token expiré, token malformé, secret manquant
- `sklearn_model_registry.py` L27, L47, L52, L57 : fichier pkl absent, modèle non chargé avant predict
- `health.py` L26-27 : cas `/ready` quand modèle non chargé

---

## 3. Performances du modèle (référence figée)

Modèle : `models/fraud_detector_v1.pkl`
Algorithme : RandomForestClassifier(n_estimators=100, random_state=42)
Données d'entraînement : 10 000 appels synthétiques (80% légitimes, 20% fraudes)

| Métrique | Classe FRAUD | Attendu (CLAUDE.md) |
|---|---|---|
| Precision | 0.94 | 0.94 |
| Recall | 0.91 | 0.91 |
| F1-score | 0.92 | 0.92 |

Ces valeurs sont les références de non-régression. Tout réentraînement doit les maintenir
ou les améliorer pour passer les tests de `tests/non_regression/`.

---

## 4. Verdict de validation

| Critère | Seuil | Résultat | Verdict |
|---|---|---|---|
| Tests passants | 100% | 48/48 | CONFORME |
| Couverture globale | ≥ 90% | 97% | CONFORME |
| Exigences F-01 à F-18 couvertes | 100% | 18/18 | CONFORME |
| Exigences NF couvertes | 100% | 3/5 | NON CONFORME (G-01, G-04) |
| Règles architecturales | 100% | 4/4 | CONFORME (vérification manuelle) |

**Statut global : VALIDÉ avec réserves**

Réserves ouvertes :
- **G-01** : SLA 50ms non vérifié par test automatique — test de charge manquant
- **G-04** : Règle anti-import sklearn dans domain/ vérifiée manuellement, non automatisée
