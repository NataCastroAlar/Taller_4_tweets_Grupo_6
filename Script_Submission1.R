rm(list = ls())

require("pacman")
library(readr)
library(rtweet)
library(tidyverse)
library(knitr)
require("pacman")
library(dplyr)
p_load("tidyverse","textir")
p_load("tm")


train <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/train.csv")
test <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/test.csv")
submision_sample <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/sample_submission.csv")


##-----------------------Limpieza de texto y tokenización---------------------##


limpiar_tokenizar <- function(text){
  # El orden de la limpieza no es arbitrario
  # Se convierte todo el text a minúsculas
  nuevo_text <- tolower(text)
  # Eliminación de páginas web (palabras que empiezan por "http." seguidas 
  # de cualquier cosa que no sea un espacio)
  nuevo_text <- str_replace_all(nuevo_text,"http\\S*", "")
  # Eliminación de signos de puntuación
  nuevo_text <- str_replace_all(nuevo_text,"[[:punct:]]", " ")
  # Eliminación de números
  nuevo_text <- str_replace_all(nuevo_text,"[[:digit:]]", " ")
  # Eliminación de espacios en blanco múltiples
  nuevo_text <- str_replace_all(nuevo_text,"[\\s]+", " ")
  # Tokenización por palabras individuales
  nuevo_text <- str_split(nuevo_text, " ")[[1]]
  # Eliminación de tokens con una longitud < 2
  nuevo_text <- keep(.x = nuevo_text, .p = function(x){str_length(x) > 1})
  return(nuevo_text)
}
#Descartamos NAs

sapply(train, function(x) sum(is.na(x)))
train <- na.omit(train)
sapply(train, function(x) sum(is.na(x)))

# Vamos a predecir !!

#----------------------Limpieza train-----------------------------------------##

# Limpieza y tokenización de los documentos de entrenamiento
train$text <- train$text %>% map(.f = limpiar_tokenizar) %>%
  map(.f = paste, collapse = " ") %>% unlist()

# Creación de la matriz documento-término
matriz_tfidf_train <- dfm(x = train$text, remove = stopwords('spanish'))

# Se reduce la dimensión de la matriz eliminando aquellos términos que 
# aparecen en menos de 5 documentos. Con esto se consigue eliminar ruido.
matriz_tfidf_train <- dfm_trim(x = matriz_tfidf_train, min_docfreq = 5)

# Conversión de los valores de la matriz a tf-idf
matriz_tfidf_train <- tfidf(matriz_tfidf_train, scheme_tf = "prop",
                            scheme_df = "inverse")

matriz_tfidf_train

##---------------------Limpieza test------------------------------------------##

# Limpieza y tokenización de los documentos de test
test$text <- test$text %>% map(.f = limpiar_tokenizar) %>%
  map(.f = paste, collapse = " ") %>% unlist()
# Identificación de las dimensiones de la matriz de entrenamiento
# Los objetos dm() son de clase S4, se accede a sus elementos mediante @
dimensiones_matriz_train <- matriz_tfidf_train@Dimnames$features
# Conversión de vector a diccionario pasando por lista
dimensiones_matriz_train <- as.list(dimensiones_matriz_train)
names(dimensiones_matriz_train) <- unlist(dimensiones_matriz_train)
dimensiones_matriz_train <- dictionary(dimensiones_matriz_train)

# Proyección de los documentos de test
matriz_tfidf_test <- dfm(x = test$text,
                         dictionary = dimensiones_matriz_train)
matriz_tfidf_test <- tfidf(matriz_tfidf_test, scheme_tf = "prop",
                           scheme_df = "inverse")

matriz_tfidf_test

all(colnames(tf_idf_test) == colnames(tf_idf))

##--------------------------Vamos a la predicción-----------------------------##

library(e1071)
modelo_svm <- svm(x = matriz_tfidf_train, y = as.factor(train$name),
                  kernel = "linear", cost = 1, scale = TRUE,
                  type = "C-classification")
modelo_svm

predicciones <- predict(object = modelo_svm, newdata = tf_idf_test)
predicciones

df_nuevo <- cbind(test, predicciones)
df_nuevo

#Submitimos:

Submission1 <- df_nuevo %>%
  select(id, predicciones)

Submission1 <- Submission1 %>%
  rename(name = predicciones)

setwd("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/Repositorios/Taller_4_tweets_Grupo_6/")

write.csv(Submission1, file="submission1.csv", row.names = F)

