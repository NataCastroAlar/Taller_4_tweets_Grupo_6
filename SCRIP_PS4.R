rm(list = ls())

require("pacman")
library(readr)

train <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/train.csv")
test <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/test.csv")
submision_sample <- read_csv("~/Desktop/MAESTRIA 2023/Big Data and Machine Learning/9. Talleres/Taller 4/Data/sample_submission.csv")

require("pacman")
p_load("tidyverse","textir")
p_load("tm")


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

#--->Train
test <- test %>% mutate(texto_tokenizado = map(.x = text,
                                                 .f = limpiar_tokenizar))
test %>% select(texto_tokenizado) %>% head()
test %>% slice(1) %>% select(texto_tokenizado) %>% pull()

##--------------------Analisis Exploratorio------------------------------------#
#Expansión ó unnest
train_tidy <- train %>% select(-text) %>% unnest()
train_tidy <- train_tidy %>% rename(token = texto_tokenizado)
head(train_tidy)

#Frecuencia de palabras

train_tidy %>% group_by(name) %>% summarise(n = n()) 

train_tidy %>%  ggplot(aes(x = name)) + geom_bar() + coord_flip() + theme_bw() 

#palabras distintas por cada usuario

train_tidy %>% select(name, token) %>% distinct() %>%  group_by(name) %>%
  summarise(palabras_distintas = n()) 

train_tidy %>% select(name, token) %>% distinct() %>%
  ggplot(aes(x = name)) + geom_bar() + coord_flip() + theme_bw()

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

##--------------------------Clasificación de Tweets---------------------------##

##PENDIENTE
??createDataPartition

p_load(caret)

set.seed(1011)
inTrain <- createDataPartition(
  y = train_tidy$name,## La variable dependiente u objetivo 
  p = .7, ## Usamos 70%  de los datos en el conjunto de entrenamiento 
  list = FALSE)

bdtrain_is <- train_tidy[ inTrain,]
bdtest_is  <- train_tidy[-inTrain,]
colnames(bdtrain_is)

table(bdtrain_is$name) / length(bdtrain_is$name)

table(bdtest_is$name) / length(bdtest_is$name)

p_load(quanteda)
# Creación de la matriz documento-término
# Creación de la matriz documento-término

token_texto <- bdtrain_is$token

matriz_tfidf_train <- dfm(token_texto)

# Se reduce la dimensión de la matriz eliminando aquellos términos que 
# aparecen en menos de 5 documentos. Con esto se consigue eliminar ruido.
matriz_tfidf_train <- dfm_trim(x = matriz_tfidf_train, min_docfreq = 5)

# Conversión de los valores de la matriz a tf-idf
matriz_tfidf_train <- tfidf(matriz_tfidf_train, scheme_tf = "prop",
                            scheme_df = "inverse")