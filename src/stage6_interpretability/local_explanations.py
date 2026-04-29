"""
Stage 6 LIME local explanations module for representative TP, FN and FP predictions.

Overall this module is where I generate per-instance explanations using LIME, the basic
idea is that global importance tells me what the model uses on average but I also need
to defend specific predictions, which is what local explanations give me. In my project
I focused on five examples here, two true positives (showing the model working), two
false negatives (showing why missed escalations were missed) and one false positive
(showing why a non-escalation got flagged). What this module demonstrates is the local
interpretability evidence the brief expects.
"""

import numpy as np
from scipy.sparse import issparse
from lime.lime_tabular import LimeTabularExplainer


def run_lime_explanations(pipeline, X_train, X_val, val_idx, df, y_prob_val,
                          y_pred_val, feature_names):
    """
    Generate LIME explanations for representative TP, FN and FP examples.

    Overall this function builds a LIME tabular explainer on my training matrix and
    then explains five carefully picked validation examples, the basic idea is to show
    the marker which words drove specific predictions for each error type. What this
    also handles is the sparse-to-dense conversion that LIME needs to work correctly.
    """
    print("\n" + "=" * 70)
    print("LOCAL EXPLANATIONS: LIME")
    print("=" * 70)

    # LIME needs dense input, so I materialise the transformed matrices once
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    X_val_transformed = pipeline.named_steps['preprocessor'].transform(X_val)

    if issparse(X_train_transformed):
        X_train_dense = X_train_transformed.toarray()
        X_val_dense = X_val_transformed.toarray()
    else:
        X_train_dense = np.array(X_train_transformed)
        X_val_dense = np.array(X_val_transformed)

    feature_names_all = list(feature_names)

    explainer = LimeTabularExplainer(
        training_data=X_train_dense,
        feature_names=feature_names_all,
        class_names=['Not Escalated', 'Escalated'],
        mode='classification',
        random_state=42,
        discretize_continuous=True
    )

    classifier = pipeline.named_steps['classifier']

    # Building a small dataframe of the validation predictions so I can pick examples
    val_df = df.iloc[val_idx].copy()
    val_df['y_prob'] = y_prob_val
    val_df['y_pred'] = y_pred_val
    val_df['prediction_type'] = 'TN'
    val_df.loc[(val_df['escalated'] == 1) & (val_df['y_pred'] == 1), 'prediction_type'] = 'TP'
    val_df.loc[(val_df['escalated'] == 0) & (val_df['y_pred'] == 1), 'prediction_type'] = 'FP'
    val_df.loc[(val_df['escalated'] == 1) & (val_df['y_pred'] == 0), 'prediction_type'] = 'FN'

    # Picking representative examples, 2 TP (most confident + median), 2 FN (highest +
    # lowest probability misses) and 1 FP (most confident false alarm)
    examples = {}
    for ptype in ['TP', 'FN', 'FP']:
        subset = val_df[val_df['prediction_type'] == ptype]
        if ptype == 'TP':
            sorted_sub = subset.sort_values('y_prob', ascending=False)
            picks = sorted_sub.iloc[[0, len(sorted_sub)//2]] if len(sorted_sub) > 1 else sorted_sub.iloc[:1]
        elif ptype == 'FN':
            sorted_sub = subset.sort_values('y_prob', ascending=False)
            picks = sorted_sub.iloc[[0, -1]] if len(sorted_sub) > 1 else sorted_sub.iloc[:1]
        else:
            sorted_sub = subset.sort_values('y_prob', ascending=False)
            picks = sorted_sub.iloc[:1]
        examples[ptype] = picks

    print("\nGenerating LIME explanations for selected examples...")

    lime_results = []
    example_counter = 0

    for ptype, picks in examples.items():
        for idx_pos, (orig_idx, row) in enumerate(picks.iterrows()):
            example_counter += 1
            val_position = list(val_df.index).index(orig_idx)

            instance = X_val_dense[val_position]
            exp = explainer.explain_instance(
                instance, classifier.predict_proba,
                num_features=10, labels=(1,)
            )

            # Handling LIME's quirk where the label key may not always be 1
            try:
                exp_list = exp.as_list(label=1)
            except KeyError:
                available_label = list(exp.local_exp.keys())[0]
                exp_list = exp.as_list(label=available_label)
                if available_label == 0:
                    exp_list = [(f, -w) for f, w in exp_list]

            print(f"\n--- Example {example_counter}: {ptype} ---")
            print(f"  Email: \"{row['email_body_text']}\"")
            print(f"  Actual: {'Escalated' if row['escalated'] == 1 else 'Not Escalated'} "
                  f"(level {row['escalation_level']})")
            print(f"  Predicted prob: {row['y_prob']:.3f} -> "
                  f"{'Flagged' if row['y_pred'] == 1 else 'Not flagged'}")
            print(f"  Top contributing features (toward escalation):")
            for feat, weight in exp_list[:10]:
                direction = "escalation" if weight > 0 else "de-escalation"
                print(f"    {feat:<50s} {weight:+.4f} ({direction})")

            lime_results.append({
                'example': example_counter,
                'type': ptype,
                'email': row['email_body_text'],
                'actual': int(row['escalated']),
                'escalation_level': int(row['escalation_level']),
                'predicted_prob': float(row['y_prob']),
                'predicted': int(row['y_pred']),
                'top_features': exp_list[:10]
            })

    return lime_results
