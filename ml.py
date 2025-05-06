y = data['Readmitted']
data=data.drop('Readmitted',axis=1)
categorical_features = data.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = data.select_dtypes(include=['number']).columns.tolist()
X = data[categorical_features + numerical_features]
# Define the preprocessing steps for categorical and numerical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')) 
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a ColumnTransformer to apply the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the classification models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(),
}

# Create a pipeline for each model
pipelines = {}
for model_name, model in models.items():
    pipelines[model_name] = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataFrame to store the evaluation metrics
results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Train and evaluate each pipeline
for model_name, pipeline in pipelines.items():
    print(f"Training {model_name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class
    recall = recall_score(y_test, y_pred, average='weighted')       
    f1 = f1_score(y_test, y_pred, average='weighted')              
    
    # Append the results to the DataFrame
    results = pd.concat([results, pd.DataFrame({
        'Model': [model_name], 
        'Accuracy': [accuracy], 
        'Precision': [precision], 
        'Recall': [recall], 
        'F1-Score': [f1]
    })], ignore_index=True)

    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))
    print("-" * 30)

# Print the results table
print("\nEvaluation Metrics:")
print(results)

print("Model training and evaluation complete.")