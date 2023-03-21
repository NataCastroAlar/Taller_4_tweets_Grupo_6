
################################################
#                                              #
# Taller Nro 4: REDES NEURONALES               #
#                                              #
################################################

# Limpiar todo

rm(list = ls())

#Librerias a utilizar
# Cargar pacman (contiene la función p_load)
library(pacman) 

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, janitor, tm, stringi, tidytext, stopwords, wordcloud2, udpipe,
       ggcorrplot)

#-------------Importamos datos-----------------

setwd("C:/Users/User/Desktop/Dulio/Big Data y Machine Learning/taller 4")

train <- read_csv("train.csv")
test <- read_csv("test.csv")

sample_submission <- read_csv("sample_submission.csv")

# Visualizamos las primeras filas
#head(train)

# Veamos el glimpse

#glimpse(train)

#table(train$text)

# Limpias de emojis 

emoji_regex <- "[\\p{So}]+" # Expresión regular para buscar emojis en cualquier categoría Unicode
test$text <- stri_replace_all_regex(test$text, emoji_regex, "bogota")

#test[test$id == 'cb9ac947c675464803342fc9',]

#Uniendo bases del train y test
#===============================================================================
# Creando columna "name" en la base test con valor "cam"

test$name <- "cam"

# Juntando bases

train =rbind(train, test)


# Limpiando datos del TRAIN
#===============================================================================
# Matriz TF-IDF

# Vamos a limpiar esta variable
# Ponemos todo el texto en minuscula
train$text[1]

train["text"] <- apply(train["text"],1,tolower)
train["text"] <- apply(train["text"],1,removeNumbers)
train["text"] <- apply(train["text"],1,removePunctuation)
train["text"] <- apply(train["text"],1,stripWhitespace)
train["text"] <- apply(train["text"],1,function(x)
  stri_trans_general(str = x, id = "Latin-ASCII"))
train$text[1]

# Transformamos el data frame a nivel de palabras
words <- train %>%
  unnest_tokens(output = "word", input = "text")

# Eliminamos stopwords
sw <- c()
for (s in c("snowball", "stopwords-iso", "nltk")) {
  temp <- get_stopwords("spanish", source = s)$word
  sw <- c(sw, temp)
}
sw <- unique(sw)
sw <- unique(stri_trans_general(str = sw, id = "Latin-ASCII"))
sw <- data.frame(word = sw)

# Número de palabras antes de remover stopwords
nrow(words)

# Remover los stopwords
words <- words %>%
  anti_join(sw, by = "word")

# Número de palabras después de remover stopwords
nrow(words)

# Veamos una nube de palabras de las 100 palabras más frecuentes
n_words <- words %>%
  count(word) %>%
  arrange(desc(n)) %>%
  head(100)

wordcloud2(data = n_words)

#========================================xxxxxxxxxxx
# Volvemos a nuestro formato original. Comentario por fila

data_clean <- words %>%
  group_by(id, name) %>% 
  summarise(comentario = str_c(word, collapse = " ")) %>%
  ungroup()

# Se eliminaron dos comentarios. Estos solo estaban compuestos por stopwords y 
# espacios.
setdiff(train$id, data_clean$id)

#===============================================================================
# Creamos un corpus
tm_corpus <- Corpus(VectorSource(x = data_clean$comentario))
str(tm_corpus)

#===============================================================================
# Creamos TF-IDF
tf_idf <- TermDocumentMatrix(tm_corpus,
                             control = list(weighting = weightTfIdf))

tf_idf <- as.matrix(tf_idf) %>%
  t() %>%
  as.data.frame()

# Revisamos que todo este ok
data_clean$comentario[1]

tf_idf[1, 1:10]

head(tf_idf)   # vemos todo su contenido

# Mega contra! Muchas columnas. 
dim(tf_idf)

# Nos vamos a quedar con las columnas que tengan los valores mas altos
columnas_seleccionadas <- colSums(tf_idf) %>%
  data.frame() %>%
  arrange(desc(.)) %>%
  head(50) %>%
  rownames()

#===============================================================================
# Dividiendo base en train y test

llaves <- data_clean %>%
  select(id, name)

todo <- cbind(llaves, tf_idf)
test_1 = todo[todo$name == "cam",]
test_1 <- test_1 %>% select(-id,-name)

train_1 = todo[todo$name != "cam",]
train_1 <- train_1 %>% select(-id,-name)

nombre <- data_clean %>%
  select(name)
Y = nombre[nombre$name != "cam",]

#===============================================================================

# Modelo

#Librerias a utilizar

#install.packages('keras')
library(keras)
library(tensorflow)
library(reticulate)

#sum(is.na(data_clean$name))

#table(data_clean$name)

# Sacamos la variable Y
Y_categotica <- factor(Y$name)
Y_categorica <- factor(Y_categotica,levels = c("Lopez", "Uribe", "Petro"), labels = c(1,2,3))
Y_categorica = as.numeric(Y_categorica)-1
Y_categotica <- to_categorical(Y_categorica,3)


X <- as.matrix(train_1)
X_test <- as.matrix(test_1)
class(X)

dim(X)

# Modelo
set.seed(1234)
model2 <- keras_model_sequential() 
# Premio para el que me diga la formula de la función de activación softmax
# y me diga que es
model2 %>% 
  layer_dense(units = 2, activation = 'relu', input_shape = ncol(X)) %>% 
  layer_dense(units = 3, activation = 'softmax')
summary(model2)

model2 %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = c('CategoricalAccuracy')
)


history <- model2 %>% 
  fit(
    X, Y_categotica, 
    epochs = 10, 
    # Truco pa la vida. El batch_size debe ser un número del estilo 2^x por motivos
    # de eficiencia computacional
    batch_size = 128,
    # Toca set pequeño de validación porque estamos jodidos de datos
    validation_split = 0.2
  )

plot(history)

#¿Que significa que la muestra de validación sea más alta que la de entrenamiento?

#model %>% evaluate(X_test, y_test)

y_hat <- model2  %>% predict(X_test) %>% k_argmax()

prediccion <- as.array(y_hat)

predid_categ <- factor(prediccion, levels = c(0,1,2), labels = c("Lopez", "Uribe", "Petro"))

name_pred = as.data.frame(predid_categ)

names(name_pred)[names(name_pred)== "predid_categ"] <- "name"

sample_submission$name = NULL

sample_submission = cbind(sample_submission, name_pred)

# Para enviar prediccion

write.csv(sample_submission, file="predichos.csv",  row.names = F)








