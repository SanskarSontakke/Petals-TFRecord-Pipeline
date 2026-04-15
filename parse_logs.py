import re
import matplotlib.pyplot as plt
import pandas as pd
import json

log_path = "./kaggle_output/petals-tfrecord-pipeline.log"

with open(log_path, 'r') as f:
    logs = f.read()

# Try to extract the final status lines for all epochs 
# Format often seen:
# categorical_accuracy: 0.0663 - loss: 4.5509 - val_categorical_accuracy: 0.0261 - val_loss: 4.6531 - learning_rate: 1.0000e-05

pattern = re.compile(r'categorical_accuracy:\s+([0-9.]+)\s+-\s+loss:\s+([0-9.]+)\s+-\s+val_categorical_accuracy:\s+([0-9.]+)\s+-\s+val_loss:\s+([0-9.]+)\s+-\s+learning_rate:\s+([0-9.eE+-]+)')

matches = pattern.findall(logs)

epochs = list(range(1, len(matches) + 1))
acc = [float(m[0]) for m in matches]
loss = [float(m[1]) for m in matches]
val_acc = [float(m[2]) for m in matches]
val_loss = [float(m[3]) for m in matches]
lr = [float(m[4]) for m in matches]

if not epochs:
    print("No matches found. Checking standard keras output.")
    pattern2 = re.compile(r'accuracy:\s+([0-9.]+).*?val_accuracy:\s+([0-9.]+)')
    # Just in case the format changed slightly
    print(logs[:1000])

df = pd.DataFrame({'Epoch': epochs, 'Accuracy': acc, 'Val_Accuracy': val_acc, 'Loss': loss, 'Val_Loss': val_loss, 'Learning_Rate': lr})
print(f"Extracted {len(epochs)} epochs.")
if len(epochs) > 0:
    print(f"Max Val Acc: {max(val_acc):.4f}")

# Plotting
dest_dir = "/home/sanskarsontakke/.gemini/antigravity/brain/6acc0f3f-82fb-4624-a717-09d88b0f7dc1/"

# 1. Accuracy Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Accuracy'], marker='o', label='Training Accuracy', color='#007acc')
plt.plot(df['Epoch'], df['Val_Accuracy'], marker='s', label='Validation Accuracy', color='#d62728')
plt.title('Training and Validation Accuracy', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Categorical Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(dest_dir + 'accuracy_plot.png', dpi=150)
plt.close()

# 2. Learning Rate Plot
plt.figure(figsize=(10, 4))
plt.plot(df['Epoch'], df['Learning_Rate'], marker='^', color='#ff7f0e', label='Learning Rate')
plt.title('Learning Rate Schedule (Exponential Decay via TPU Scheduler)', fontsize=14, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.yscale('log')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(dest_dir + 'lr_plot.png', dpi=150)
plt.close()

with open(dest_dir + "metrics.json", "w") as f:
    json.dump({
        "max_val_acc": max(val_acc) if val_acc else 0,
        "max_acc": max(acc) if acc else 0
    }, f)
print("Plots generated and saved.")
