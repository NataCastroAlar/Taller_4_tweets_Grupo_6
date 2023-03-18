rm(list = ls())

# Cargar pacman (contiene la función p_load)
library(pacman) 

# Cargar las librerías listadas e instalarlas en caso de ser necesario
p_load(tidyverse, janitor, tm, stringi, tidytext, stopwords, wordcloud2, udpipe,
       ggcorrplot) 

data <- read.csv("https://github.com/ignaciomsarmiento/datasets/raw/main/tripadvisor_medellin.csv", sep = ";")
glimpse(data)

data <- clean_names(data)
names(data)

# Cantidad de restaurantes
length(unique(data$nombre))

# Cantidad de comentarios por restaurante
n_comentarios = data %>%
  group_by(nombre) %>%
  summarise(n = n()) %>%
  ungroup()

ggplot(n_comentarios, aes(x = n)) +
  geom_histogram(fill = "darkblue", alpha = 0.7) +
  theme_bw() +
  labs(x = "Número de reseñas por restaurante", y = "Cantidad")

n_comentarios %>%
  arrange(desc(n)) %>%
  head()

data %>%
  distinct(nombre, estrellas_restaurante) %>%
  ggplot(aes(x = estrellas_restaurante)) +
  geom_bar(fill = "darkblue", alpha = 0.7) +
  theme_bw() +
  labs(x = "Estrellas por restaurante", y = "Cantidad")

# En general, la mayoría de restaurantes están top. Ahora veamos los comentarios
ggplot(data, aes(x = calificacion)) +
  geom_bar(fill = "darkblue", alpha = 0.7) +
  theme_bw() +
  labs(x = "Estrellas por comentario", y = "Cantidad")

# Vamos a comenzar creando nuestra variable titulo + comentario
data["full_text"] = paste(data$titulo_comentario, data$comentario)


# Eliminamos las columnas que no vamos a usar
data = select(data, -titulo_comentario, -comentario)


# Vamos a limpiar esta variable
# Ponemos todo el texto en minuscula
data["full_text"] <- apply(data["full_text"], 1, tolower)
# Eliminamos numeros
data["full_text"] <- apply(data["full_text"], 1, removeNumbers)
# Eliminamos signos de puntuación
data["full_text"] <- apply(data["full_text"], 1, removePunctuation)
# Eliminamos multiples espacios en blanco
data["full_text"] <- apply(data["full_text"], 1, stripWhitespace)
# Elimiamos acentos
data["full_text"] <- apply(data["full_text"], 1, function(x) 
  stri_trans_general(str = x, id = "Latin-ASCII"))

# Tenemos que transformar nuestro dataframe para que quede a nivel de palabra,
# sin embargo necesitamos crear un id de comentario para no perder el rastro de 
# nuestra unidad de observación. 

data <- data %>%
  mutate(id = row_number())

words <- data %>%
  unnest_tokens(output = "word", input = "full_text")

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

# Vamos a lematizar

#udpipe_download_model("spanish")
#udpipe::udpipe_download_model('spanish')

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
  tail(100)

# Para que la matriz TF-IDF no sea GIGANTE vamos a eliminar todas las palabras que aparezcan menos de 10 veces. ¿Por qué 10? No sé, me la estoy jugando, con esto pueden jugar ustedes después
palabras_eliminar <- words %>%
  count(lemma) %>%
  filter(n < 10)

words <- words %>%
  anti_join(palabras_eliminar, by = "lemma") 

# Volvemos a nuestro formato original. Comentario por fila
data_clean <- words %>%
  group_by(nombre, estrellas_restaurante, usuario, calificacion, id) %>% 
  summarise(comentario = str_c(lemma, collapse = " ")) %>%
  ungroup()

# Se eliminaron dos comentarios. Estos solo estaban compuestos por stopwords y 
# espacios.
setdiff(data$id, data_clean$id)

data[c(1490, 1491),]

# Creamos un corpus
tm_corpus <- Corpus(VectorSource(x = data_clean$comentario))
str(tm_corpus)

# Creamos TF-IDF
tf_idf <- TermDocumentMatrix(tm_corpus,
                             control = list(weighting = weightTfIdf))


tf_idf <- as.matrix(tf_idf) %>%
  t() %>%
  as.data.frame()

# Revisamos que todo este ok
data_clean$comentario[1]

tf_idf[1, 1:10]

head(tf_idf)

# Mega contra! Muchas columnas. 
dim(tf_idf)

# Nos vamos a quedar con las columnas que tengan los valores mas altos
columnas_seleccionadas <- colSums(tf_idf) %>%
  data.frame() %>%
  arrange(desc(.)) %>%
  head(50) %>%
  rownames()

tf_idf_reducido <- tf_idf %>%
  select(all_of(columnas_seleccionadas))

#Salvamos datos
save(data, data_clean, tf_idf, tf_idf_reducido, 
     file = "/Users/victorsanchez/Desktop/MAESTRIA 2023/Big Data and Machine Learning/Repositorios/Taller_4_tweets_Grupo_6/")
