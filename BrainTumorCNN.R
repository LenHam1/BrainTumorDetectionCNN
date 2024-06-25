library(keras)


# Seed für Reprouzierbarkeit
set.seed(42)


# Pfad zu training und testing datensatz
train_dir <- "pfad_zu_trainingsdaten"
test_dir <- "pfad_zu_testdaten"


# Zielgröße für Bilder festlegen, Channel = 3 für rgb
img_width <- 150
img_height <- 150
channels <- 3


# Preprocessing der Bilder
preprocess_image <- function(image_path) {
  image <- keras::image_load(image_path, target_size = c(img_width, img_height))
  image <- image_to_array(image) / 255
  image <- array_reshape(image, c(1, img_width, img_height, channels))
  return(image)
}


# Daten vorbereiten
prepare_data <- function(directory) {
  categories <- list.files(directory, full.names = TRUE)
  images <- lapply(categories, function(cat) list.files(cat, full.names = TRUE, pattern = "\\.jpg$", recursive = TRUE))
  labels <- rep(seq_along(categories) - 1, sapply(images, length))
  images <- unlist(images, recursive = FALSE)
  x <- lapply(images, preprocess_image)
  y <- as.numeric(factor(labels)) - 1
  return(list(x = array_reshape(do.call(c, x), c(length(images), img_width, img_height, channels)), y = y))
}


# prepare_data Funktion für Trainings- & Tesdaten aufrufen
train_data <- prepare_data(train_dir)
test_data <- prepare_data(test_dir)


# Anzahl der Kategorien der Trainings- & Testdaten kontrolieren
table(train_data$y)
table(test_data$y)


# CNN Modell erstellen
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(4, 4), activation = "relu", input_shape = c(img_width, img_height, channels)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(4, 4), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(4, 4), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>% # um Overfitting zu minimieren --> zufällig 50% der Filter ausschalten
  layer_dense(units = 4, activation = "softmax")


# Modell compilen
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy")
)



# Modell trainieren
history <- model %>% fit(
  train_data$x, train_data$y,
  epochs = 11,
  batch_size = 20, # so viele Bilder werden verarbeitet bevor Parameter des Modells angepasst werden
  validation_split = 0.1
)


# Auswertung des Modells für Testdaten
evaluation <- model %>% evaluate(test_data$x, test_data$y)


# Accuracy ausgeben
evaluation <- as.list(evaluation)
accuarcy <- cat("Test accuracy:", evaluation$acc * 100, "%\n")

