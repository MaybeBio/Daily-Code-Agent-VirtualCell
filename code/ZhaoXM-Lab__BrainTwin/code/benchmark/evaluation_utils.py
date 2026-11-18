import json
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def SCQ_metrics(true_labels, predicted_labels, show=False):
    """Score multiple-choice labels. Returns accuracy, precision, recall, and F1."""
    cm = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    if show:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['A', 'B', 'C', 'D', 'other'], 
                   yticklabels=['A', 'B', 'C', 'D', 'other'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)

def load_question_data(file_path):
    """Load question records and their labels."""
    with open(file_path, "r") as f:
        question_list = json.load(f)
    
    labels = [question['output'] for question in question_list]
    return question_list, labels


def evaluate_agent_on_dataset(agent, llm, question_list, labels, show=False):
    result_list = []
    record_list = []
    for question in tqdm(question_list):
        try:
            result = agent.invoke({
        "system_prompts":question['system'],
        "instruction_prompts": question['instruction'],
        "question": question['input'],
    })
            result_list.append(result)
            if result['intermediate_steps']==[]:
                record_list.append("without tool")
            else:
                record_list.append("with tool")
        except:
            result = llm.invoke(f"system: {question['system']}\ninstruction: {question['instruction']}\nquestion: {question['input']}").content
            result_list.append(result)
            record_list.append("failed")


    predictions = []
    for result in result_list:
        if type(result)==dict:
            result = result['output']
        if result not in ['A', 'B', 'C', 'D']:
            # Try to find A, B, C, or D anywhere in the result, not just at the beginning
            match = re.search(r'([A-D])', result)
            if match:
                predictions.append(match.group(1))
            else:
                # If no A-D found, append a default value or skip
                print(f"Warning: No A-D found in result: {result[:100]}...")
                predictions.append('A')  # Default fallback
        else:
            predictions.append(result)
            
    print((sum(np.array(record_list)=="without tool")+sum(np.array(record_list)=="with tool"))/len(record_list)*100, "% successed")
    print(sum(np.array(record_list)=="with tool")/len(record_list)*100, "% used tools")
    print(SCQ_metrics(labels, predictions, show))
    
    return result_list, record_list