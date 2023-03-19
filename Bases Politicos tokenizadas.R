# RANDOM FOREST################----------
###LIMPIEZA Y MATRICES GRUPO 6#########
#Siguiendo clase complementaria

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

sw <- c()
for (s in c("snowball", "stopwords-iso", "nltk")) {
  temp <- get_stopwords("spanish", source = s)$word
  sw <- c(sw, temp)
}
sw <- unique(sw)
sw <- unique(stri_trans_general(str = sw, id = "Latin-ASCII"))
#Creo data frame con stopwords
sw <- data.frame(word = sw)

# Número de palabras antes de remover stopwords
nrow(words)
## Elimino stopwords
words <- words %>%
  anti_join(sw, by = "word")

# Número de palabras después de remover stopwords
nrow(words)

#n_words <- words %>%
#count(word) %>%
#arrange(desc(n)) %>%
#head(100)

#wordcloud2(data = n_words)

# Vamos a lematizar , descargamos modelo en español
udpipe::udpipe_download_model('spanish')
model <- udpipe_load_model(file = "spanish-gsd-ud-2.5-191206.udpipe")
palabras_unicas <- words %>%
  distinct(word)
udpipe_results <- udpipe_annotate(model, x = palabras_unicas$word)
udpipe_results <- as_tibble(udpipe_results)
udpipe_results <- udpipe_results %>% 
  select(token, lemma) %>%
  rename("word" = "token")
words <- words %>%
  left_join(udpipe_results, by = "word", multiple = "all")
words[is.na(words$lemma), "lemma"] <- words[is.na(words$lemma), "word"]

# Veamos las palabras menos comunes
words %>%
  count(lemma) %>%
  arrange(desc(n)) %>%
  tail(10)

#Palabras lematizando
length(unique(words$lemma))

# Elimnamos palabras que aparezcan menos de 10 veces
palabras_eliminar <- words %>%
  count(lemma) %>%
  filter(n < 10)

words <- words %>%
  anti_join(palabras_eliminar, by = "lemma") 


# Volvemos a nuestro formato original. Comentario por fila
data_clean <- words %>%
  group_by(name, lemma, id) %>% 
  summarise(comentario = str_c(lemma, collapse = " ")) %>%
  ungroup()

#setdiff(data$id, data_clean$id)

# Creamos un corpus
tm_corpus <- Corpus(VectorSource(x = data_clean$comentario))
str(tm_corpus)

# Creamos TF-IDFpara obtener la matriz con pesos
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

#data_clean$name <- factor(data_clean$name)
#data_clean$name <-as.numeric(data_clean$name)


save(data, data_clean, tf_idf, tf_idf_reducido, file= "datos_para_modelar_G63.RData")

############



### CREAMOS TRAIN Y TEST--------------------------------------------------------

## Creamos la matriz que vamos a utilizar la cual debe incluir la variable a predecir en este caso name

#Nos quedamos sólo con el vector name
name<-data_clean%>%
  select(name=name)


#La pegamos a la matriz dtm y data.frame


matriz_df<-cbind(name, tf_idf_reducido)
class(matriz_df)


###MATRIZ DATA FRAME
set.seed(123)
train_index <- createDataPartition(matriz_df$name, p = .8)$Resample1
train_df <- matriz_df[train_index,]
test_df <- matriz_df[-train_index,]




###MODELO RANDOM FOREST---------------------------------------------------------

mtry_grid<-expand.grid(mtry =c(10,20, 25,30, 40, 45, 50))
ctrl<- trainControl(method = "cv",
                    number = 7)


p_load(randomForest)
set.seed(123)
forest <- train(name~., 
                data = train_df, 
                method = "rf",
                trControl = ctrl,
                tuneGrid=mtry_grid,
                metric="Accuracy",
                ntree=30
)

forest

plot(forest)

y_hat_insample1 = predict(forest, newdata = train_df)
y_hat_outsample1 = predict(forest, newdata = test_df)
-----------------------------
 