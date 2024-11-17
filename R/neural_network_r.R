# Load necessary libraries
library(keras)

# Generate example data
set.seed(123)
X <- matrix(runif(100), nrow=10, ncol=10)
y <- sample(0:1, 10, replace=TRUE)

# Define the model
model <- keras_model_sequential() %>%
  layer_dense(units = 32, activation = 'relu', input_shape = c(10)) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(X, y, epochs = 10, batch_size = 2)

# Evaluate the model
model %>% evaluate(X, y)
