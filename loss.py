import numpy as np

def get_classes(category_tokens, category='category'):
    """Get sorted class labels for this category."""
    okc_vals = category_tokens.get('okc_vals')
    if not okc_vals:
        print(f"No classes found in category tokens for {category}.")
        return None
    classes = sorted(okc_vals.keys())
    return classes

def np_arrays(guesses, category_tokens, category):
    """Format guesses and ground truth in a form suitable for numpy operations."""
    okc_vals = category_tokens.get('okc_vals')
    classes = get_classes(category_tokens, category)
    ground_truth = []
    munged_guesses = []
    for guess in guesses:
        current_ground_truth = guess.get(category, {}).get('ground_truth')
        if not current_ground_truth:
            continue
        current_guesses = [(float(guess.get(category, {}).get('estimate').get(clas, '0%')[:-1]) / 100) for clas in classes]
        # print(f'Guesses: {current_guesses}. {current_ground_truth}') # XXX
        ground_truth.append([1 if okc_vals.get(clas) == current_ground_truth else 0 for clas in classes])
        munged_guesses.append(current_guesses)
    if len(munged_guesses) == 0:
        return None, None # non-existent categories
    ground_truth = np.array(ground_truth)
    munged_guesses = np.array(munged_guesses)
    return ground_truth, munged_guesses

def ce_loss(guesses, category_tokens, category):
    """Calculate cross-entropy loss given probabilities and ground truth."""
    classes = get_classes(category_tokens, category)
    ground_truth, munged_guesses = np_arrays(guesses, category_tokens, category)
    if ground_truth is None or munged_guesses is None:
        return None
    print(f'Sanity check: {ground_truth.shape} == {munged_guesses.shape}') # XXX
    loss = -np.sum(ground_truth * np.log(munged_guesses + 1e-9)) / len(ground_truth)  # Added epsilon (1e-9) to prevent log(0)
    print("Cross-entropy loss:", loss) # XXX
    return loss

def generate_confusion_matrix(guesses, category_tokens, category):
    # Using the np_arrays function to obtain predictions and ground truth
    ground_truth, munged_guesses = np_arrays(guesses, category_tokens, category)
    if ground_truth is None:
        print(f"No ground truth for category {category}.")
        return None
    if munged_guesses is None:
        print(f"Couldn't create munged_guesses for category {category}.")
        return None
    # Find the index of the maximum prediction probability for each case
    predicted_classes = np.argmax(munged_guesses, axis=1)
    actual_classes = np.argmax(ground_truth, axis=1)
    
    # Obtain number of classes from the shape of the ground truth array
    num_classes = ground_truth.shape[1]
    
    # Compute the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(actual_classes, predicted_classes):
        confusion_matrix[actual, predicted] += 1
    
    return confusion_matrix