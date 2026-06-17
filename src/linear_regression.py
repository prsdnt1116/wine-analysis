import pandas as pd
import numpy as np


df = pd.read_csv("data/wine.csv")

target = "quality"
features = [
    "alcohol",
    "volatile acidity",
    "sulphates",
    "citric acid",
    "density",
]

X = df[features]
y = df[target]

random_state = np.random.default_rng(42)
indices = random_state.permutation(len(df))
test_size = int(len(df) * 0.2)
test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train = X.iloc[train_indices].to_numpy()
X_test = X.iloc[test_indices].to_numpy()
y_train = y.iloc[train_indices].to_numpy()
y_test = y.iloc[test_indices].to_numpy()

X_train_with_bias = np.column_stack([np.ones(len(X_train)), X_train])
X_test_with_bias = np.column_stack([np.ones(len(X_test)), X_test])
weights = np.linalg.lstsq(X_train_with_bias, y_train, rcond=None)[0]
y_pred = X_test_with_bias @ weights

mse = np.mean((y_test - y_pred) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("features:", features)
print("coefficients:", dict(zip(features, [float(value) for value in weights[1:]])))
print("intercept:", float(weights[0]))
print("MSE:", float(mse))
print("R2:", float(r2))

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual quality")
    plt.ylabel("Predicted quality")
    plt.title("Wine Quality Linear Regression")
    plt.tight_layout()
    plt.savefig("linear_regression_result.png", dpi=200)
    plt.show()
except ModuleNotFoundError:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 900, 700
    margin_left, margin_bottom = 100, 90
    margin_right, margin_top = 50, 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_min, x_max = float(y_test.min()), float(y_test.max())
    y_min, y_max = float(y_pred.min()), float(y_pred.max())
    axis_min = min(x_min, y_min)
    axis_max = max(x_max, y_max)
    padding = (axis_max - axis_min) * 0.08
    axis_min -= padding
    axis_max += padding

    def to_pixel(actual, predicted):
        x = margin_left + (actual - axis_min) / (axis_max - axis_min) * plot_width
        y = margin_top + (axis_max - predicted) / (axis_max - axis_min) * plot_height
        return x, y

    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_width, margin_top + plot_height],
        outline="black",
    )
    line_start = to_pixel(axis_min, axis_min)
    line_end = to_pixel(axis_max, axis_max)
    draw.line([line_start, line_end], fill="red", width=3)

    for actual, predicted in zip(y_test, y_pred):
        x, yy = to_pixel(float(actual), float(predicted))
        draw.ellipse([x - 3, yy - 3, x + 3, yy + 3], fill="#2f6fbb")

    draw.text((width // 2 - 130, 25), "Wine Quality Linear Regression", fill="black", font=font)
    draw.text((width // 2 - 45, height - 40), "Actual quality", fill="black", font=font)
    draw.text((15, height // 2), "Predicted quality", fill="black", font=font)
    draw.text((margin_left, height - 65), f"{axis_min:.1f}", fill="black", font=font)
    draw.text((margin_left + plot_width - 30, height - 65), f"{axis_max:.1f}", fill="black", font=font)
    draw.text((55, margin_top + plot_height - 8), f"{axis_min:.1f}", fill="black", font=font)
    draw.text((55, margin_top - 8), f"{axis_max:.1f}", fill="black", font=font)

    image.save("linear_regression_result.png")
