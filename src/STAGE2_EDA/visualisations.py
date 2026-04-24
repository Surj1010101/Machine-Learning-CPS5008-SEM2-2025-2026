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
