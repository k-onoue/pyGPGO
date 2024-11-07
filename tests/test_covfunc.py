import torch
import numpy as np
from _src import squaredExponential, matern32, matern52

# 使用するカーネル関数のインスタンス
covfuncs = [squaredExponential(), matern32(), matern52()]

# 勾配計算が可能なカーネル
grad_enabled = [squaredExponential(), matern32(), matern52()]

# カーネル名とクラスのマッピング
covariance_classes = {
    "squaredExponential": squaredExponential,
    "matern32": matern32,
    "matern52": matern52,
}

# ハイパーパラメータの範囲
hyperparameters_interval = {
    "squaredExponential": {"l": (0.1, 2.0), "sigmaf": (0.1, 1.0), "sigman": (0.0, 0.1)},
    "matern32": {"l": (0.1, 2.0), "sigmaf": (0.1, 1.0), "sigman": (0.0, 0.1)},
    "matern52": {"l": (0.1, 2.0), "sigmaf": (0.1, 1.0), "sigman": (0.0, 0.1)},
}


def generate_hyperparameters(**hyperparameter_interval):
    generated_hyperparameters = {}
    for hyperparameter, bound in hyperparameter_interval.items():
        generated_hyperparameters[hyperparameter] = np.random.uniform(
            bound[0], bound[1]
        )
    return generated_hyperparameters


def test_psd_covfunc():
    # 生成されたカーネルが正定値であることを確認
    torch.manual_seed(0)
    np.random.seed(0)
    for name in covariance_classes:
        for i in range(10):
            hyperparams = generate_hyperparameters(**hyperparameters_interval[name])
            cov = covariance_classes[name](**hyperparams)
            for j in range(10):
                X = torch.randn(10, 2)
                K = cov.K(X, X)
                # 対称性を確保
                K = (K + K.T) / 2
                # 最小固有値をチェック
                eigvals = torch.linalg.eigvalsh(K)
                assert (
                    eigvals > -1e-6
                ).all(), f"{name} カーネルの共分散行列が正定値ではありません"


def test_sim():
    # カーネル関数のシミュレーションテスト
    torch.manual_seed(0)
    X = torch.randn(100, 3)
    for cov in covfuncs:
        K = cov.K(X, X)
        assert K.shape == (
            100,
            100,
        ), f"{cov.__class__.__name__} カーネルの出力サイズが不正です"


def test_grad():
    # 勾配計算のテスト
    torch.manual_seed(0)
    X = torch.randn(3, 3, requires_grad=True)
    for cov in grad_enabled:
        K = cov.K(X, X)
        K_sum = K.sum()
        K_sum.backward()
        grad = X.grad
        assert (
            grad is not None
        ), f"{cov.__class__.__name__} カーネルで勾配計算に失敗しました"


if __name__ == "__main__":
    test_psd_covfunc()
    test_sim()
    test_grad()
    print("すべてのテストが成功しました。")
