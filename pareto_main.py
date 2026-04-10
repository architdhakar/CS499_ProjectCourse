from dataset import load_adult
from pareto.pareto_pipeline import ParetoPipeline
import random

print("="*60)
print("PARETO OPTIMISATION PIPELINE")
print("="*60)

train_dataset, test_dataset, formatter, label_fn = load_adult()

train_data = [train_dataset[i] for i in range(len(train_dataset))]
test_data = [test_dataset[i] for i in range(len(test_dataset))]

random.shuffle(train_data)

split = int(len(train_data)*0.3)
val_data = train_data[split:]
train_data = train_data[:split]

pipeline = ParetoPipeline(train_data, val_data, formatter, label_fn, k=4)

pipeline.run()

selected = pipeline.select()

print(f"\nSelected {len(selected)} demos (Pareto)")