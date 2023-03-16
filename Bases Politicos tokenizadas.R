rm(list = ls())
require("pacman")
library(readr)

require("pacman")

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, # Manipular dataframes
       tm,   # para Text Mining
       tidytext, #Para tokenización
       wordcloud, # Nube de palabras 
       SentimentAnalysis #Análisis de sentimientos 
) 


train <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/train.csv")
test <- read_csv("Documents/UNIANDES BIG DATA MACHINE LEARNING/Archivos R/DATOS TALLERES/DATOS POLITICOS/test.csv")


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


##---------------------------------MATRIZ--------------------------------##

###CREAMOS MATRIZ UTILIZANDO COUNT VECTORIZER

comentarios <- train_tidy$token

#Quitamos números, signos de puntuación etc
comentarios <- removeNumbers(comentarios)
comentarios <- removePunctuation(comentarios)
comentarios <- tolower(comentarios)
comentarios <- stripWhitespace(comentarios)
comentarios <- iconv(comentarios, from = "UTF-8", to="ASCII//TRANSLIT")

texts<-VCorpus(VectorSource(comentarios))
texts
matriz_1<-DocumentTermMatrix(texts)
matriz_1
inspect(matriz_1[1:50, 20:40])
ncol(matriz_1)







