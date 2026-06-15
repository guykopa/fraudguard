# Plan de test — fraudguard

## Objectif

Ce document formalise la stratégie de test du service d'inférence fraudguard,
définit les niveaux de vérification appliqués, et établit la traçabilité entre
les exigences de CLAUDE.md et les tests qui les vérifient.

---

## Niveaux de test (modèle en V)

```
EXIGENCES SYSTÈME (CLAUDE.md)          ←→   NON-RÉGRESSION
  CONCEPTION ARCHITECTURALE (Architecture.md)  ←→   INTÉGRATION
    CONCEPTION DÉTAILLÉE (interfaces/)         ←→   UNITAIRES
              IMPLÉMENTATION (code)
```

| Niveau | Répertoire | Scope | Fakes autorisés |
|---|---|---|---|
| Unitaire | `tests/unit/` | Une classe isolée | Oui — FakeModelRegistry, FakeFeatureExtractor |
| Intégration | `tests/integration/` | Flux HTTP complet, modèle réel | Non |
| Non-régression | `tests/non_regression/` | Comportement figé du modèle pkl | Non |

---

## Matrice de traçabilité — exigences → tests

### Exigences fonctionnelles

| ID | Exigence (CLAUDE.md) | Test(s) couvrant |
|---|---|---|
| F-01 | Extraction de 6 features depuis CallEvent | `test_extracts_six_features`, `test_feature_names_order` |
| F-02 | `duration_normalized` = duration_sec / 3600 ∈ [0,1] | `test_duration_normalized_value`, `test_duration_normalized_between_0_and_1` |
| F-03 | `is_night_call` = 1 si hour_of_day ∈ {0,1,2,3,4,5} | `test_is_night_call_true_at_3am`, `test_is_night_call_false_at_2pm` |
| F-04 | `is_international` = 1 si destination ne commence pas par +33 | `test_is_international_true_for_non_french_destination`, `test_is_international_false_for_french_destination` |
| F-05 | `calls_last_hour`, `sms_last_hour`, `unique_dest_24h` passés bruts | `test_calls_last_hour_raw`, `test_sms_last_hour_raw`, `test_unique_dest_24h_raw` |
| F-06 | risk_score ≥ 0.7 → FRAUD | `test_returns_fraud_for_high_risk_score`, `test_fraud_call_always_classified_as_fraud` |
| F-07 | risk_score ≥ 0.4 → SUSPICIOUS | `test_suspicious_label_for_mid_risk_score` |
| F-08 | risk_score < 0.4 → LEGITIMATE | `test_legitimate_call_returns_legitimate_label`, `test_normal_call_always_classified_as_legitimate` |
| F-09 | PSI < 0.1 → DriftStatus.OK | `test_no_drift_on_identical_distributions` |
| F-10 | PSI ∈ [0.1, 0.2[ → DriftStatus.WARNING | `test_warning_drift_on_moderately_different_distributions` |
| F-11 | PSI ≥ 0.2 → DriftStatus.CRITICAL | `test_critical_drift_on_very_different_distributions` |
| F-12 | Division par zéro dans PSI gérée sans exception | `test_handles_zero_division_gracefully` |
| F-13 | Réponse `/api/v1/predict` contient label, risk_score, model_version, latency | `test_predict_fraud_call`, `test_predict_legitimate_call`, `test_predict_returns_model_version` |
| F-14 | `/api/v1/predict` requiert JWT valide | `test_predict_requires_auth` |
| F-15 | `/api/v1/model/info` retourne version et 6 feature_names | `test_model_info` |
| F-16 | `/api/v1/monitoring/drift` retourne 6 rapports PSI | `test_drift_endpoint` |
| F-17 | `/health` retourne `{"status": "ok"}` | `test_health_liveness` |
| F-18 | `/ready` retourne `{"status": "ready"}` quand modèle chargé | `test_health_readiness` |

### Exigences non-fonctionnelles

| ID | Exigence (CLAUDE.md) | Test(s) couvrant | Statut |
|---|---|---|---|
| NF-01 | Latence d'inférence mesurée et exposée sur chaque prédiction | `test_measures_inference_latency`, `test_predict_legitimate_call` (latency > 0) | VÉRIFIÉ |
| NF-02 | Latence < 50ms en production | `test_inference_latency_under_50ms` (seuil CI : 500ms) | PARTIEL — pas de test de charge |
| NF-03 | Couverture de test ≥ 90% | Gate CI `--cov-fail-under=90` | VÉRIFIÉ (97% actuel) |
| NF-04 | JWT_SECRET depuis variable d'environnement | `jwt_handler.py` + test d'intégration (`os.environ.setdefault`) | VÉRIFIÉ |
| NF-05 | Modèle chargé uniquement via IModelRegistry | Convention architecturale + règle mypy | NON TESTÉ automatiquement |

### Exigences architecturales (SOLID / Hexagonal)

| ID | Règle | Vérification |
|---|---|---|
| A-01 | Pas d'import sklearn dans `domain/` ou `interfaces/` | `grep -r "sklearn" fraudguard/domain fraudguard/interfaces` → vide |
| A-02 | Ports sans logique métier | Revue de code — interfaces contiennent uniquement `@abstractmethod` |
| A-03 | Injection par constructeur dans tous les use cases | `PredictionService.__init__`, `DriftService.__init__` |
| A-04 | Seul `main.py` instancie des adaptateurs concrets | Convention vérifiée à la revue |

---

## Données de test

### Cas canoniques (fixtures `conftest.py`)

| Fixture | Description | Usage |
|---|---|---|
| `normal_call` | Appel domestique, journée, courte durée | Tests unitaires feature engineering + prediction |
| `fraud_call` | Appel international, nuit, longue durée, volume élevé | Tests unitaires + non-régression |
| `night_call` | hour_of_day = 2, domestic | Test `is_night_call` isolé |
| `reference_features` | Distribution normale μ=2.0, σ=0.5, n=1000, seed=42 | Tests drift baseline |
| `drifted_features` | Distribution normale μ=8.0, σ=1.5, n=1000, seed=42 | Tests drift critique |

### Adaptateurs de test (fakes)

| Classe | Remplace | Comportement |
|---|---|---|
| `FakeModelRegistry` | `SklearnModelRegistry` | Retourne `fraud_proba` fixe, pas de fichier pkl |
| `FakeFeatureExtractor` | `PandasFeatureExtractor` | Retourne un FeatureVector fixe en μs |

---

## Critères de validation

| Critère | Seuil | Résultat actuel |
|---|---|---|
| Tous les tests passent | 100% | 48/48 PASSED |
| Couverture globale | ≥ 90% | 97% |
| Aucune erreur mypy | 0 erreurs | À vérifier en CI |
| Aucune erreur flake8 | 0 erreurs | À vérifier en CI |
| Latence non-régression | < 500ms (CI) | PASSED |

---

## Lacunes identifiées et actions

| ID | Lacune | Priorité | Action recommandée |
|---|---|---|---|
| G-01 | SLA 50ms non vérifié automatiquement | Haute | Ajouter un test de charge (locust) avec assertion p99 < 50ms |
| G-02 | Cas-limite SUSPICIOUS (risk_score ∈ [0.4, 0.7[) non couvert en non-régression | Moyenne | Ajouter un `CallEvent` entraînant risk_score ~0.55 |
| G-03 | Chemins d'erreur `jwt_handler.py` et `sklearn_model_registry.py` non couverts (81-86%) | Moyenne | Tests des cas `ModelNotFoundError`, token expiré |
| G-04 | Règle A-01 (pas de sklearn dans domain/) vérifiée manuellement | Basse | Ajouter un test pytest qui grep les imports interdits |
