import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_visualisations(df, corr):
    """Generate and save all Stage 2 plots."""
    print("\n" + "="*70)
    print("GENERATING VISUALISATIONS...")
    print("="*70)

    #maain overview figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Stage 2: EDA Overview', fontsize=14, fontweight='bold')

    #plot 1:Target distribution
    df['escalated'].value_counts().plot(kind='bar', ax=axes[0, 0],
                                        color=['#2196F3', '#F44336'])
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_xticklabels(['Not Escalated (0)', 'Escalated (1)'], rotation=0)
    axes[0, 0].set_ylabel('Count')

    # Plot 2:Sentiment vs escalation
    sent_esc = df.dropna(subset=['sentiment']).groupby('sentiment')['escalated'].mean()
    sent_esc.plot(kind='bar', ax=axes[0, 1], color='#FF9800')
    axes[0, 1].set_title('Escalation Rate by Sentiment')
    axes[0, 1].set_ylabel('Escalation Rate')
    axes[0, 1].tick_params(axis='x', rotation=0)

        # Plot 3: Issue category vs escalation
    issue_esc = df.groupby('issue_category')['escalated'].mean().sort_values(ascending=False)
    issue_esc.plot(kind='bar', ax=axes[0, 2], color='#4CAF50')
    axes[0, 2].set_title('Escalation Rate by Issue Category')
    axes[0, 2].set_ylabel('Escalation Rate')
    axes[0, 2].tick_params(axis='x', rotation=45)

    # Plot 4: Text length distribution
    for esc_val, color, label in [(0, '#2196F3', 'Not Escalated'),
                                   (1, '#F44336', 'Escalated')]:
        axes[1, 0].hist(df[df['escalated'] == esc_val]['text_len'], bins=30,
                        alpha=0.6, color=color, label=label)
    axes[1, 0].set_title('Text Length Distribution')
    axes[1, 0].set_xlabel('Character Length')
    axes[1, 0].legend()

    # Plot 5: Emotion intensity distribution
    for esc_val, color, label in [(0, '#2196F3', 'Not Escalated'),
                                   (1, '#F44336', 'Escalated')]:
        axes[1, 1].hist(df[df['escalated'] == esc_val]['emotion_intensity'], bins=30,
                        alpha=0.6, color=color, label=label)
    axes[1, 1].set_title('Emotion Intensity Distribution')
    axes[1, 1].set_xlabel('Emotion Intensity')
    axes[1, 1].legend()

    # Plot 6: Sentiment missingness
    miss_data = pd.DataFrame({
        'Not Escalated': [df[(df['escalated'] == 0) & df['sentiment'].notna()].shape[0],
                          df[(df['escalated'] == 0) & df['sentiment'].isna()].shape[0]],
        'Escalated': [df[(df['escalated'] == 1) & df['sentiment'].notna()].shape[0],
                      df[(df['escalated'] == 1) & df['sentiment'].isna()].shape[0]]
    }, index=['Present', 'Missing'])
    miss_data.T.plot(kind='bar', stacked=True, ax=axes[1, 2],
                     color=['#4CAF50', '#F44336'])
    axes[1, 2].set_title('Sentiment Missingness by Escalation')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].tick_params(axis='x', rotation=0)



    


