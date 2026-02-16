import matplotlib.pyplot as plt

def plot_results(results):
    names = [r["name"] for r in results]
    fairness = [r["fairness"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    gap = [r["gap"] for r in results]

    # Fairness vs Accuracy
    plt.figure(figsize=(6,5))
    plt.scatter(fairness, accuracy)
    for i, name in enumerate(names):
        plt.annotate(name, (fairness[i], accuracy[i]))
    plt.xlabel("Fairness (Entropy)")
    plt.ylabel("Accuracy")
    plt.title("Fairness vs Accuracy")
    plt.show()

    # Accuracy vs Demographic Gap
    plt.figure(figsize=(6,5))
    plt.scatter(accuracy, gap)
    for i, name in enumerate(names):
        plt.annotate(name, (accuracy[i], gap[i]))
    plt.xlabel("Accuracy")
    plt.ylabel("Gender Accuracy Gap")
    plt.title("Accuracy vs Fairness Gap")
    plt.show()

    # Bar plot comparison
    plt.figure(figsize=(8,4))
    plt.bar(names, accuracy)
    plt.title("Accuracy Comparison")
    plt.show()
