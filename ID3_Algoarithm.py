import math

def calculate_entropy(data, target_col):
    value_counts = {}
    total = len(data)
    for row in data:
        target_value = row[target_col]
        if target_value not in value_counts:
            value_counts[target_value] = 0
        value_counts[target_value] += 1
    entropy = 0
    for count in value_counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_information_gain(data, attribute, target_col):
    total_entropy = calculate_entropy(data, target_col)
    attribute_values = set(row[attribute] for row in data)
    weighted_entropy = 0
    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_entropy = calculate_entropy(subset, target_col)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    info_gain = total_entropy - weighted_entropy
    return info_gain

def build_tree(data, attributes, target_col):
    target_values = [row[target_col] for row in data]
    if len(set(target_values)) == 1:
        return target_values[0]
    if not attributes:
        return max(set(target_values), key=target_values.count)
    info_gains = {attribute: calculate_information_gain(data, attribute, target_col) for attribute in attributes}
    best_attribute = max(info_gains, key=info_gains.get)
    tree = {best_attribute: {}}
    remaining_attributes = [attribute for attribute in attributes if attribute != best_attribute]
    best_attribute_values = set(row[best_attribute] for row in data)
    for value in best_attribute_values:
        subset = [row for row in data if row[best_attribute] == value]
        tree[best_attribute][value] = build_tree(subset, remaining_attributes, target_col)
    return tree

def predict(tree, data_point):
    if isinstance(tree, dict):
        attribute = next(iter(tree))
        value = data_point[attribute]
        return predict(tree[attribute][value], data_point)
    else:
        return tree

def test_decision_tree():
    data = [
        {'Weather': 'Sunny', 'Temperature': 'Hot', 'Play?': 'No'},
        {'Weather': 'Overcast', 'Temperature': 'Hot', 'Play?': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 'Mild', 'Play?': 'Yes'},
        {'Weather': 'Sunny', 'Temperature': 'Cool', 'Play?': 'Yes'},
        {'Weather': 'Sunny', 'Temperature': 'Hot', 'Play?': 'No'},
        {'Weather': 'Overcast', 'Temperature': 'Cool', 'Play?': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 'Mild', 'Play?': 'Yes'},
        {'Weather': 'Rainy', 'Temperature': 'Hot', 'Play?': 'No'}
    ]
    attributes = ['Weather', 'Temperature']
    target_col = 'Play?'
    tree = build_tree(data, attributes, target_col)
    print("Decision Tree:")
    print(tree)
    new_data_point = {'Weather': 'Sunny', 'Temperature': 'Mild'}
    prediction = predict(tree, new_data_point)
    print(f"Prediction for {new_data_point}: {prediction}")

test_decision_tree()
