import shap

def tree_explainer(model, X):
    explainer = shap.TreeExplainer(model)
    return explainer(X)
