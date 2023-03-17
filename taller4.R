
################################################
#                                              #
# Taller Nro 4 / MODELOS                       #
#                                              #
################################################

# Limpiar todo

rm(list = ls())

#Librerias a utilizar

require("pacman")
library(readr)
library(tidyverse)
p_load("stringi")
p_load("tm")
library(tensorflow)
#-------------Importamos datos-----------------

setwd("C:/Users/User/Desktop/Dulio/Big Data y Machine Learning/taller 4")

train <- read_csv("train.csv")
test <- read_csv("test.csv")

sample_submission <- read_csv("sample_submission.csv")

# Visualizamos las primeras filas
head(train)

# Veamos el glimpse

glimpse(train)

table(train$text)

# Limpiando texto
train$text[1]

comentarios <- stri_trans_general(str = train$text, id = "Latin-ASCII")
comentarios[1]
comentarios <- iconv(comentarios, from = "UTF-8", to="ASCII//TRANSLIT")
comentarios[1]

texts<-VCorpus(VectorSource(comentarios))
texts

texts<-tm_map(texts,content_transformer(tolower))
texts[[1]]$content
texts<-tm_map(texts,content_transformer(removePunctuation))
texts[[1]]$content
texts<-tm_map(texts,content_transformer(removeNumbers))
texts[[1]]$content
texts<-tm_map(texts,content_transformer(tolower))
texts[[1]]$content


p_load("stopwords")
# Descargamos la lista de las stopwords en español de dos fuentes diferentes y las combinamos
lista_palabras1 <- stopwords(language = "es", source = "snowball")
lista_palabras2 <- stopwords(language = "es", source = "nltk")
lista_palabras <- union(lista_palabras1, lista_palabras2)

texts<-tm_map(texts,removeWords,lista_palabras)
texts[[1]]$content
texts<-tm_map(texts,content_transformer(stripWhitespace))
texts[[1]]$content

# Matriz de Terminos del Documento (DTM)
matriz_1<-DocumentTermMatrix(texts)
matriz_1
inspect(matriz_1[1,])

matriz_1 <- removeSparseTerms(matriz_1, sparse = 0.95)
matriz_1
inspect(matriz_1[1:2,])
dim(matriz_1)

# DTM usando el vectorizador Tf-IDF
dtm_idf_texts <- DocumentTermMatrix(texts, control = list(weighting=weightTfIdf))
inspect(dtm_idf_texts[1:2,])
dim(dtm_idf_texts)

# variables
train_matrix <- as.matrix(matriz_1)

train_nombre <- factor(train$name, levels = unique(train$name))
train_nombre <- to_categorical(as.integer(train_nombre) - 1)

# Modelo 
# Armando la primera red
library(keras)
library(reticulate)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = dim(train_matrix)[2]) %>% 
  layer_dense(units = 3, activation = 'softmax')

summary(model)

# Entrenando la primera red
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)


history <- model %>% fit(
  train_matrix, train_nombre, 
  epochs = 30, 
  batch_size = 60,
  validation_split = 0.2
)

plot(history)


#model %>% evaluate(x_test, y_test)

## Test
#=========

# Limpiando texto
test$text[1]

comentarios1 <- stri_trans_general(str = test$text, id = "Latin-ASCII")
comentarios1[1]
comentarios1 <- iconv(comentarios1, from = "UTF-8", to="ASCII//TRANSLIT")
comentarios1[1]

texts1<-VCorpus(VectorSource(comentarios1))
texts1

texts1<-tm_map(texts1,content_transformer(tolower))
texts1[[1]]$content
texts1<-tm_map(texts1,content_transformer(removePunctuation))
texts1[[1]]$content
texts1<-tm_map(texts1,content_transformer(removeNumbers))
texts1[[1]]$content
texts1<-tm_map(texts1,content_transformer(tolower))
texts1[[1]]$content


p_load("stopwords")
# Descargamos la lista de las stopwords en español de dos fuentes diferentes y las combinamos
lista_palabras1 <- stopwords(language = "es", source = "snowball")
lista_palabras2 <- stopwords(language = "es", source = "nltk")
lista_palabras <- union(lista_palabras1, lista_palabras2)

texts1<-tm_map(texts1,removeWords,lista_palabras)
texts1[[1]]$content
texts1<-tm_map(texts1,content_transformer(stripWhitespace))
texts1[[1]]$content

# Matriz de Terminos del Documento (DTM)
matriz_2<-DocumentTermMatrix(texts1)
matriz_2
inspect(matriz_2[1,])

matriz_2 <- removeSparseTerms(matriz_2, sparse = 0.95)
matriz_2
inspect(matriz_2[1:2,])
dim(matriz_2)

# DTM usando el vectorizador Tf-IDF
dtm_idf_texts1 <- DocumentTermMatrix(texts1, control = list(weighting=weightTfIdf))
inspect(dtm_idf_texts1[1:2,])
dim(dtm_idf_texts1)

# variables
test_matrix <- as.matrix(matriz_2)
test_matrix <- cbind(test_matrix, matrix(0, nrow = nrow(test_matrix), ncol = 1))

#========
model  %>% predict(test_matrix) %>% k_argmax()

predicciones_categorias <- categorias[model]

# Agregar una nueva columna al data.frame de prueba con las categorías correspondientes
test_df$categorias_predicciones <- predicciones_categorias

# Mostrar el data.frame actualizado
print(test_df)

