import json
import joblib
import src.train as train

from src.train import make_pipeline, evaluate


def test_make_pipeline():
    model = make_pipeline()

    assert "scaler" in model.named_steps
    assert "model" in model.named_steps


def test_evaluate():
    model = make_pipeline()

    X_train = [[1, 2], [2, 3], [8, 9], [9, 10]]
    y_train = [0, 0, 1, 1]

    X_test = [[1.5, 2.5], [8.5, 9.5]]
    y_test = [0, 1]

    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_log_run(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path)

    logged_metrics = {}

    class DummyRunInfo:
        run_id = "test-run-123"

    class DummyRun:
        info = DummyRunInfo()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(train.mlflow, "start_run", lambda run_name: DummyRun())
    monkeypatch.setattr(train.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda metrics: logged_metrics.update(metrics))
    monkeypatch.setattr(train.mlflow, "log_artifact", lambda path: None)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda model, artifact_path: None)
    monkeypatch.setattr(train.ConfusionMatrixDisplay, "from_estimator", lambda *args, **kwargs: None)

    X_train = [[1, 2], [2, 3], [8, 9], [9, 10]]
    y_train = [0, 0, 1, 1]
    X_test = [[1.5, 2.5], [8.5, 9.5]]
    y_test = [0, 1]

    result = train.log_run(
        "test_model",
        train.make_pipeline(),
        X_train,
        y_train,
        X_test,
        y_test,
        {"model": "LogisticRegression"}
    )

    assert result["run_id"] == "test-run-123"
    assert result["name"] == "test_model"
    assert "accuracy" in result
    assert "f1" in logged_metrics


def test_main(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "ARTIFACT_DIR", tmp_path)

    monkeypatch.setattr(
        train,
        "split_dataset",
        lambda: ([[1, 2], [2, 3]], [[3, 4]], [0, 1], [1])
    )

    class FakeSearch:
        def __init__(self, estimator, *args, **kwargs):
            self.best_estimator_ = estimator
            self.best_params_ = {"model__C": 1.0}
            self.best_score_ = 0.9

        def fit(self, X, y):
            return self

    monkeypatch.setattr(train, "GridSearchCV", FakeSearch)
    monkeypatch.setattr(train, "RandomizedSearchCV", FakeSearch)

    scores = {
        "baseline_logistic_regression": 0.80,
        "grid_search_best": 0.95,
        "random_search_best": 0.85
    }

    def fake_log_run(name, model, *args):
        return {
            "name": name,
            "f1": scores[name],
            "accuracy": scores[name],
            "precision": scores[name],
            "recall": scores[name]
        }

    monkeypatch.setattr(train, "log_run", fake_log_run)

    saved = {}
    monkeypatch.setattr(
        joblib,
        "dump",
        lambda model, path: saved.update({"path": path})
    )

    train.main()

    assert "path" in saved
    assert (tmp_path / "results.json").exists()

    results = json.loads((tmp_path / "results.json").read_text())
    assert len(results) == 3