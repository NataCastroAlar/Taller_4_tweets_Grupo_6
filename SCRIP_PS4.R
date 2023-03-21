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


train <- read_csv("~/Victor Ivan/Universidad/Taller 4/Data/train.csv")
test <- read_csv("~/Victor Ivan/Universidad/Taller 4/Data/test.csv")
submision_sample <- read_csv("~/Victor Ivan/Universidad/Taller 4/Data/sample_submission.csv")


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

#Aplicamos función propia de limpieza y tokenización
#--->Train
train <- train %>% mutate(texto_tokenizado = map(.x = text,
                                                 .f = limpiar_tokenizar))
train %>% select(texto_tokenizado) %>% head()
train %>% slice(1) %>% select(texto_tokenizado) %>% pull()

#--->Test
test <- test %>% mutate(texto_tokenizado = map(.x = text,
                                                 .f = limpiar_tokenizar))
test %>% select(texto_tokenizado) %>% head()
test %>% slice(1) %>% select(texto_tokenizado) %>% pull()

##--------------------Analisis Exploratorio------------------------------------#
#Expansión ó unnest
train_tidy <- train %>% dplyr::select(-text) %>% unnest()
train_tidy <- train_tidy %>% rename(token = texto_tokenizado)
head(train_tidy)

#Frecuencia de palabras

train_tidy %>% group_by(name) %>% summarise(n = n()) 

train_tidy %>%  ggplot(aes(x = name)) + 
  geom_bar(fill = "#386641") + 
  labs(y = 'No Palabras', 
       x = "Usuario", 
       title = "Total de Palabras por usuario") +
  coord_flip() + 
  scale_y_continuous(labels = scales::number, expand = c(0, 0)) +
  theme_bw() 

#palabras distintas por cada usuario

train_tidy %>% dplyr::select(name, token) %>% distinct() %>%  group_by(name) %>%
  summarise(palabras_distintas = n()) 

train_tidy %>% dplyr::select(name, token) %>% 
  distinct() %>%
  ggplot(aes(x = name)) + 
  geom_bar(fill = "#386641") +
  labs(y = 'Palabras distintas', 
       x = "Usuario", 
       title = "Palabras distintas por usuario") +
  coord_flip() + 
  scale_y_continuous(labels = scales::number, expand = c(0, 0)) +
  theme_minimal()

#Palabras mas usadas por usuario

train_tidy %>% group_by(name, token) %>% count(token) %>% group_by(name) %>%
  top_n(10, n) %>% arrange(name, desc(n)) %>% print(n=30)


##---------------------------------Stop Words---------------------------------##

#Eliminamos Stop Words
p_load(tm)
head(stopwords('spanish'))

train_tidy <- train_tidy  %>% 
  anti_join(tibble(token =stopwords("spanish")))

dim(train_tidy)
#Verificamos de nuevo palabras mas usadas
train_tidy %>% group_by(name, token) %>% count(token) %>% group_by(name) %>%
  top_n(10, n) %>% arrange(name, desc(n)) %>% print(n=30)

#Representación grafica de frecuencias

train_tidy %>% group_by(name, token) %>% count(token) %>% group_by(name) %>%
  top_n(10, n) %>% arrange(name, desc(n)) %>%
  ggplot(aes(x = reorder(token,n), y = n, fill = name)) +
  geom_col() +
  theme_bw() +
  labs(y = "", x = "") +
  theme(legend.position = "none") +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~name,scales = "free", ncol = 1, drop = TRUE)

##---------------------------------Word Clouds--------------------------------##

library(wordcloud)
library(RColorBrewer)

wordcloud_custom <- function(grupo, df){
  print(grupo)
  wordcloud(words = df$token, freq = df$frecuencia,
            max.words = 400, random.order = FALSE, rot.per = 0.35,
            colors = brewer.pal(8, "Dark2"))
}

df_grouped <- train_tidy %>% group_by(name, token) %>% count(token) %>%
  group_by(name) %>% mutate(frecuencia = n / n()) %>%
  arrange(name, desc(frecuencia)) %>% nest() 

walk2(.x = df_grouped$name, .y = df_grouped$data, .f = wordcloud_custom)


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



##---------------------------Predicción---------------------------------------##

install.packages("SparseM")
library(e1071)

modelo_svm <- svm(x = matriz_tfidf_train, y = as.factor(train$name),
                  kernel = "linear", cost = 1, scale = TRUE,
                  type = "C-classification")
modelo_svm

predicciones <- predict(object = modelo_svm, newdata = matriz_tfidf_test)
predicciones

##-----------------------Optimizacion de Hiperparametros----------------------##

set.seed(369)
svm_cv <- tune("svm", train.x =  matriz_tfidf_train,
               train.y = as.factor(as.factor(train$name)),
               kernel = "linear", 
               ranges = list(cost = c(0.1, 0.5, 1, 2.5, 5)))
summary(svm_cv)

ggplot(data = svm_cv$performances, aes(x = cost, y = error)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(ymin = error - dispersion, ymax = error + dispersion)) +
  theme_bw()

svm_cv$best.parameters

modelo_svm <- svm(x = matriz_tfidf_train, y = as.factor(as.factor(train$name)),
                  kernel = "linear", cost = 0.1, scale = TRUE)

predicciones2 <- predict(object = modelo_svm, newdata = matriz_tfidf_test)
predicciones2


df_nuevo <- cbind(test, predicciones2)
df_nuevo

#Submitimos:

Submission2 <- df_nuevo %>%
  select(id, predicciones2)

Submission2 <- Submission2 %>%
  rename(name = predicciones2)

setwd("~/Victor Ivan/Universidad/Taller 4/Data/")

write.csv(Submission2, file="submission2.csv", row.names = F)


##-----------------------Kernel no lineales-----------------------------------##

svm_cv_radial <- tune("svm", train.x =  matriz_tfidf_train,
                      train.y = as.factor(as.factor(train$name)), 
                      kernel = 'radial',
                      ranges = list(cost = c(0.1, 0.5, 1, 2.5, 5)),
                      gamma = c(0.5, 1, 2, 3, 4, 5, 10))

svm_cv_radial 


ggplot(data = svm_cv_radial$performances, aes(x = cost, y = error, color = as.factor(gamma)))+
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")


