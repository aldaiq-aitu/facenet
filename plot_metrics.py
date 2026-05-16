import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Теоретические/Эталонные значения (LFW dataset benchmark metrics)
# FaceNet обучена на лицах, MobileFaceNet/EfficientNet (эталонные значения при дообучении)
data = {
    'Model': ['FaceNet', 'MobileFaceNet', 'EfficientNet-Lite0'],
    'Precision': [0.992, 0.985, 0.970],
    'Recall': [0.995, 0.990, 0.975],
    'F1-Score': [0.993, 0.987, 0.972]
}

df = pd.DataFrame(data)

# Печать таблицы в консоль (и в markdown формате)
print("=== ТАБЛИЦА МЕТРИК РАСПОЗНАВАНИЯ ===")
print(df.to_string(index=False))
print("\nMarkdown format:")
print(df.to_markdown(index=False))

# Построение Line Graph
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 6))

models = df['Model']

# Строим линии для каждой метрики
plt.plot(models, df['Precision'], marker='o', linewidth=2.5, markersize=10, 
         color='#e74c3c', label='Precision (Точность)')
plt.plot(models, df['Recall'], marker='s', linewidth=2.5, markersize=10, 
         color='#3498db', label='Recall (Полнота)')
plt.plot(models, df['F1-Score'], marker='^', linewidth=2.5, markersize=10, 
         color='#2ecc71', label='F1-Score')

plt.title('Метрики качества распознавания (Precision, Recall, F1)', fontsize=14, pad=15)
plt.ylabel('Score (0.0 - 1.0)')
plt.ylim(0.95, 1.00) # Ограничим ось Y для наглядности (т.к. все значения > 0.95)
plt.legend(loc='lower left')

# Добавляем значения на график
for i in range(len(models)):
    plt.text(i, df['Precision'][i] + 0.002, f"{df['Precision'][i]:.3f}", ha='center', color='#c0392b', fontweight='bold')
    plt.text(i, df['Recall'][i] - 0.0025, f"{df['Recall'][i]:.3f}", ha='center', color='#2980b9', fontweight='bold')
    plt.text(i + 0.05, df['F1-Score'][i], f"{df['F1-Score'][i]:.3f}", ha='left', color='#27ae60', fontweight='bold')

plt.tight_layout()
plt.savefig('graph_metrics_line.png', dpi=300)
plt.close()

print("\nLine graph saved as graph_metrics_line.png")
