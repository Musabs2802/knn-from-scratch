# K-Nearest Neighbors (KNN) from Scratch

## What is KNN?

K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for both classification and regression. In KNN, the output of a query point is determined based on the majority class or average value of its 'k' nearest neighbors in the feature space. The algorithm does not make any assumptions about the underlying data distribution.

## Properties of KNN

1. **Instance-based Learning**: KNN is an instance-based learning algorithm, meaning it memorizes the training instances which are used to make predictions.
2. **Lazy Learning**: KNN is considered a lazy learner because it does not build an explicit model during the training phase. Instead, it defers computation until the prediction phase.
3. **Non-parametric**: KNN does not assume a specific form for the underlying data distribution, making it a non-parametric method.
4. **Distance Metric**: The effectiveness of KNN heavily depends on the choice of distance metric (e.g., Euclidean, Manhattan, Minkowski).
5. **Parameter 'k'**: The number of nearest neighbors ('k') is a crucial hyperparameter that affects the performance of the algorithm.

## Assumptions of KNN

1. **Similarity Assumption**: KNN assumes that similar instances exist in close proximity to each other in the feature space.
2. **Feature Relevance**: It is assumed that all features contribute equally to the distance calculation, which might not always be the case.
3. **Sufficient Data**: KNN performs better with a large amount of labeled data since it relies on instance-based learning.

## How to Calculate KNN

### KNN for Classification

1. **Determine the Value of 'k'**: Choose the number of nearest neighbors 'k'.
2. **Calculate Distance**: Compute the distance between the query point and all points in the training set using a distance metric (e.g., Euclidean distance).

$$\
   d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
   \$$
   
4. **Identify Neighbors**: Identify the 'k' nearest neighbors to the query point based on the calculated distances.
5. **Majority Voting**: Assign the class label that is most frequent among the 'k' nearest neighbors.
6. **Tie-breaking**: In case of a tie, use strategies such as reducing 'k' by one, or selecting the nearest neighbor.

### KNN for Regression

1. **Determine the Value of 'k'**: Choose the number of nearest neighbors 'k'.
2. **Calculate Distance**: Compute the distance between the query point and all points in the training set using a distance metric.

$$\
   d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
   \$$
   
4. **Identify Neighbors**: Identify the 'k' nearest neighbors to the query point based on the calculated distances.
5. **Average of Neighbors**: Compute the average of the target values of the 'k' nearest neighbors.

$$\
   \hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i
   \$$
   
6. **Prediction**: Use the computed average as the predicted value for the query point.

## When to Use KNN?

1. **Small Datasets**: KNN performs well on small datasets where the computation cost is manageable.
2. **Non-linear Data**: KNN does not assume a linear relationship between the features, making it suitable for complex, non-linear data.
3. **Instance-based Learning**: When you want to delay the generalization until a query is made, making it a lazy learner.
4. **Multi-class Classification**: Effective for problems involving multiple classes.
5. **No Assumptions**: When you do not want to make any assumptions about the underlying data distribution.

## Advantages of KNN

1. **Simplicity**: Easy to understand and implement.
2. **No Training Phase**: Since KNN is a lazy learner, there is no explicit training phase, making the implementation straightforward.
3. **Versatile**: Can be used for both classification and regression problems.
4. **Non-parametric**: Does not assume any specific distribution for the data, making it flexible.
5. **Adaptable**: Can naturally handle multi-class problems and data with multiple dimensions.

## Disadvantages of KNN

1. **Computationally Expensive**: The algorithm becomes slow and computationally expensive as the size of the dataset grows because it computes the distance to every other instance.
2. **Memory Intensive**: Requires storing all the data, which can be impractical for large datasets.
3. **Sensitive to Irrelevant Features**: The presence of irrelevant or redundant features can negatively impact the performance of KNN.
4. **Sensitive to Scaling**: KNN requires proper feature scaling (e.g., normalization or standardization) since it relies on distance calculations.
5. **Curse of Dimensionality**: Performance degrades as the number of dimensions increases, making it less effective for high-dimensional data.
6. **Imbalanced Data**: KNN can struggle with imbalanced datasets where some classes are underrepresented.
