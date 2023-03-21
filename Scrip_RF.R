# 
###RANDOM FOREST GRUPO 6######### 


rm(list = ls())

#Librerias a utilizar

require("pacman")
library(readr)
library(tidyverse)
p_load(tidyverse, janitor, tm, stringi, tidytext, stopwords, wordcloud2, udpipe,
              ggcorrplot)

#-------------Importamos datos-----------------


train <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/train.csv")
test <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/test.csv")



# Eliminamos las columnas que no vamos a usar
#data = select(train, -name, -id)
data<-train
# Vamos a limpiar esta variable 
# Ponemos todo el texto en minuscula
data["text"] <- apply(data["text"], 1, tolower)
# Eliminamos numeros
data["text"] <- apply(data["text"], 1, removeNumbers)
# Eliminamos signos de puntuación
data["text"] <- apply(data["text"], 1, removePunctuation)
# Eliminamos multiples espacios en blanco
data["text"] <- apply(data["text"], 1, stripWhitespace)
# Elimiamos acentos
data["text"] <- apply(data["text"], 1, function(x) 
  stri_trans_general(str = x, id = "Latin-ASCII"))


# Tenemos que transformar nuestro dataframe para que quede a nivel de palabra,
# sin embargo necesitamos crear un id de comentario para no perder el rastro de 
# nuestra unidad de observación. 

data <- data %>%
  mutate(id = row_number())

words <- data %>%
  unnest_tokens(output = "word", input = "text")

# Eliminamos stopwords
sw<-stopwords("es", "snowball")
sw <- unique(stri_trans_general(str = sw, id = "Latin-ASCII"))
sw<-data.frame(word=sw)

head(sw)

words <- words %>%
  anti_join(sw, by = "word")


# Número de palabras después de remover stopwords
nrow(words)


#wordcloud2(data = n_words)

# Vamos a lematizar , descargamos modelo en español
udpipe::udpipe_download_model('spanish')
model<-udpipe_load_model(file="spanish-gsd-ud-2.5-191206.udpipe")

palabras_unicas <- words %>%
  distinct(word)

results <- udpipe_annotate(model, x = palabras_unicas$word)
results <- as_tibble(results)

results <- results %>% 
  select(token, lemma) %>%
  rename("word" = "token")

words <- words %>%
  left_join(results, by = "word", multiple = "all")
words[is.na(words$lemma), "lemma"] <- words[is.na(words$lemma), "word"]

# Veamos las palabras menos comunes
words %>%
  count(lemma) %>%
  arrange(desc(n)) %>%
  head()

#Palabras lematizando
length(unique(words$lemma))

# Elimnamos palabras que aparezcan menos de 10 veces
palabras_eliminar <- words %>%
  count(lemma) %>%
  filter(n < 10)

words <- words %>%
  anti_join(palabras_eliminar, by = "lemma") 


##### Volvemos a nuestro formato original. Comentario por fila
data_clean <- words %>%
  group_by(name, lemma, id) %>% 
  summarise(comentario = str_c(lemma, collapse = " ")) %>%
  ungroup()

#setdiff(data$id, data_clean$id)

# Creamos un corpus
tm_corpus <- Corpus(VectorSource(x = data_clean$comentario))
str(tm_corpus)

# Creamos TF-IDFpara obtener la matriz con pesos###############PROBLEMA########
##############################################################################



tf_idf <- TermDocumentMatrix(tm_corpus,
                             control = list(weighting = weightTfIdf))





tf_idf <- as.matrix(tf_idf) %>%
  t() %>% #trasponemos
  as.data.frame()



# Revisamos que todo este ok
#############data_clean$comentario[1]
colnames(tf_idf)

dim(tf_idf)
#Nos quedamos con las columnas con los valores más altos
columnas_seleccionadas <- colSums(tf_idf) %>% 
  data.frame() %>%
  arrange(desc(.)) %>%
  head(50) %>%
  rownames()

tf_idf_reducido <- tf_idf %>%
  select(all_of(columnas_seleccionadas))


dim(tf_idf_reducido)



#Dónde estamos:
getwd()


data_clean$name <- as.factor(data_clean$name)
data_clean$name <-as.numeric(data_clean$name)


save(data, data_clean, tf_idf, tf_idf_reducido, file= "datos_para_modelar_G64.RData")

############

### CREAMOS TRAIN Y TEST--------------------------------------------------------

## Creamos la matriz que vamos a utilizar la cual debe incluir la variable a predecir en este caso name

#Nos quedamos sólo con el vector name
name<-data_clean%>%
  select(name=name)

#name_n<-name%>%
  #mutate(name=factor(name, levels=c(1,2,3), labels=c("petro", "lopez", "uribe")))

#La pegamos a la matriz dtm y data.frame

p_load(rpart, caret)
matriz_df<-cbind(name, tf_idf_reducido)
class(matriz_df)



###MATRIZ DATA FRAME
set.seed(123)
train_index <- createDataPartition(matriz_df$name, p = .8)$Resample1
train_df <- matriz_df[train_index,]
test_df <- matriz_df[-train_index,]




###MODELO RANDOM FOREST---------------------------------------------------------

p_load(randomForest)
set.seed(123)

mtry_grid<-expand.grid(mtry =c(7, 10))

ctrl<- trainControl(method = "cv",
                    number = 5)



forest <- train(name~., 
                data = train_df, 
                method = "rf",
                trControl = ctrl,
                tuneGrid=mtry_grid,
                metric="Accuracy"
)

forest

plot(forest)

y_hat_insample1 = predict(forest, newdata = train_df)
y_hat_outsample1 = predict(forest, newdata = test_df)
-----------------------------
p_load("SuperLearner")
listWrappers()
p_load("caret" )
set.seed(1011)

ySL<-data_clean$name
XSL<- tf_idf_reducido 
sl.lib <- c("SL.randomForest", "SL.glmnet") 

fitY <- SuperLearner(Y = ySL,  X= data.frame(XSL),
                     method = "method.NNLS", # combinación convexa
                     SL.library = sl.lib)

fitY

