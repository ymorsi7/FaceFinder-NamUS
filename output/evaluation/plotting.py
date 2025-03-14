import matplotlib.pyplot as plt
import re

def extract_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        precision = float(re.search(r'Precision: (\d+\.\d+)%', content).group(1)) # regex
        recall = float(re.search(r'Recall: (\d+\.\d+)%', content).group(1))
        f1 = float(re.search(r'F1 Score: (\d+\.\d+)%', content).group(1))
        accuracy = float(re.search(r'Accuracy: (\d+\.\d+)%', content).group(1))
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1, 
            'Accuracy': accuracy
        }

def plot_metrics(metrics):
    plt.figure(figsize=(10, 6))
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.bar(names, values)
    plt.ylim(0, 100)
    plt.title('Face Matching Performance')
    plt.ylabel('Percentage (%)')
    
    # labels over bars
    for i, v in enumerate(values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.savefig('output/evaluation/performance.png')
    plt.close()

if __name__ == "__main__":
    metrics = extract_metrics('output/evaluation/evaluation_summary.txt')
    plot_metrics(metrics)