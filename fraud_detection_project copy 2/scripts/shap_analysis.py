import pandas as pd
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

X_train = pd.read_csv('data/X_train_balanced.csv')
X_test = pd.read_csv('data/X_test.csv')

xgb_model = XGBClassifier()
xgb_model.load_model('models/xgb_model.json')

explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/4_shap_summary_plot.png")

try:
    plt.figure(figsize=(15, 5))
    shap.force_plot(
        explainer.expected_value[0],
        shap_values[1],
        X_test.iloc[1],
        matplotlib=True
    ).savefig("outputs/4_shap_force_plot.png")
except Exception as e:
    print(f"Error generating SHAP force plot: {e}")

shap.dependence_plot("Amount", shap_values.values, X_test, show=False)
plt.savefig("outputs/4_shap_dependence_plot.png")
