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
    plt.xlabel("実際の品質")
    plt.ylabel("予測した品質")
    plt.title("ワイン品質の線形回帰: 実際の値と予測値")
    plt.tight_layout()
    plt.savefig("linear_regression_result.png", dpi=200)
    plt.show()
except ModuleNotFoundError:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 1000
    margin_left, margin_bottom = 150, 140
    margin_right, margin_top = 90, 130
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/YuGothB.ttc", 32)
        label_font = ImageFont.truetype("C:/Windows/Fonts/YuGothB.ttc", 24)
        text_font = ImageFont.truetype("C:/Windows/Fonts/YuGothM.ttc", 20)
        small_font = ImageFont.truetype("C:/Windows/Fonts/YuGothM.ttc", 18)
    except OSError:
        title_font = label_font = text_font = small_font = ImageFont.load_default()

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
        [0, 0, width - 1, height - 1],
        outline="#d0d7de",
    )

    ticks = np.linspace(np.floor(axis_min), np.ceil(axis_max), 7)
    for tick in ticks:
        x_tick, _ = to_pixel(float(tick), axis_min)
        _, y_tick = to_pixel(axis_min, float(tick))

        draw.line(
            [(x_tick, margin_top), (x_tick, margin_top + plot_height)],
            fill="#e5e7eb",
            width=1,
        )
        draw.line(
            [(margin_left, y_tick), (margin_left + plot_width, y_tick)],
            fill="#e5e7eb",
            width=1,
        )
        draw.line(
            [(x_tick, margin_top + plot_height), (x_tick, margin_top + plot_height + 10)],
            fill="black",
            width=2,
        )
        draw.line(
            [(margin_left - 10, y_tick), (margin_left, y_tick)],
            fill="black",
            width=2,
        )
        draw.text(
            (x_tick - 15, margin_top + plot_height + 18),
            f"{tick:.1f}",
            fill="#24292f",
            font=small_font,
        )
        draw.text(
            (margin_left - 60, y_tick - 7),
            f"{tick:.1f}",
            fill="#24292f",
            font=small_font,
        )

    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_width, margin_top + plot_height],
        outline="black",
        width=2,
    )
    line_start = to_pixel(axis_min, axis_min)
    line_end = to_pixel(axis_max, axis_max)
    draw.line([line_start, line_end], fill="#d1242f", width=4)

    for actual, predicted in zip(y_test, y_pred):
        x, yy = to_pixel(float(actual), float(predicted))
        draw.ellipse([x - 5, yy - 5, x + 5, yy + 5], fill="#0969da", outline="#054da7")

    draw.text(
        (margin_left, 45),
        "ワイン品質の線形回帰: 実際の品質と予測した品質",
        fill="#24292f",
        font=title_font,
    )
    draw.text(
        (margin_left, 88),
        "説明変数: アルコール度数, 揮発酸, 硫酸塩, クエン酸, 密度",
        fill="#57606a",
        font=text_font,
    )
    draw.text(
        (margin_left + plot_width - 260, margin_top + 30),
        f"R2 = {r2:.3f}",
        fill="#24292f",
        font=text_font,
    )
    draw.text(
        (margin_left + plot_width - 260, margin_top + 55),
        f"MSE = {mse:.3f}",
        fill="#24292f",
        font=text_font,
    )
    draw.line(
        [
            (margin_left + plot_width - 260, margin_top + 95),
            (margin_left + plot_width - 210, margin_top + 95),
        ],
        fill="#d1242f",
        width=4,
    )
    draw.text(
        (margin_left + plot_width - 200, margin_top + 87),
        "理想線",
        fill="#24292f",
        font=text_font,
    )
    draw.text(
        (width // 2 - 90, height - 70),
        "実際の品質",
        fill="#24292f",
        font=label_font,
    )
    y_label = Image.new("RGBA", (260, 45), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label)
    y_label_draw.text((0, 0), "予測した品質", fill="#24292f", font=label_font)
    image.paste(y_label.rotate(90, expand=True), (35, height // 2 - 130), y_label.rotate(90, expand=True))

    image.save("linear_regression_result.png")
